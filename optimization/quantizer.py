"""4-bit / 8-bit compression of Linear layers.

Prefers bitsandbytes NF4 / Linear8bitLt when the wheel actually imports.
On CPU CI and most laptops it does not, so we fall back to a fake-quant
wrapper that still swaps the modules and reports the byte delta. The
fallback is not bitwise-compatible with bnb; do not ship those weights
to a GPU runtime and expect them to load.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("omni.quant")

QuantMode = Literal["nf4", "int8", "none"]


class QuantError(RuntimeError):
    pass


def bitsandbytes_status() -> dict[str, Any]:
    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception as exc:  # ImportError *and* the CUDA runtime surprises
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "reason": "import_ok"}


class FakeLinear4Bit(nn.Module):
    """Group-wise NF4-ish fake quant. Enough to catch wrap bugs without GPU."""

    def __init__(self, lin: nn.Linear, group: int = 64) -> None:
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.group = group
        w = lin.weight.detach()
        q, scale = _group_nf4(w, group)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _dequant_nf4(self.qweight, self.scale)[:, : self.in_features]
        return F.linear(x, w, self.bias)


class FakeLinear8Bit(nn.Module):
    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        w = lin.weight.detach()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.float() * self.scale
        return F.linear(x, w, self.bias)


def _group_nf4(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    out, inn = w.shape
    pad = (group - (inn % group)) % group
    if pad:
        w = F.pad(w, (0, pad))
    g = w.view(out, -1, group)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    # 16-level symmetric quant, close enough to NF4 for profiling
    q = torch.round((g / scale) * 7).clamp(-8, 7).to(torch.int8)
    return q, scale.squeeze(-1)


def _dequant_nf4(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out, ngrp, group = q.shape
    g = q.float() / 7.0 * scale.unsqueeze(-1)
    w = g.reshape(out, ngrp * group)
    return w


def _replace_linears(
    module: nn.Module,
    factory: Any,
    *,
    skip: Iterable[str] = (),
) -> int:
    skipped = set(skip)
    swapped = 0
    for name, child in list(module.named_children()):
        if name in skipped:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, factory(child))
            swapped += 1
        else:
            swapped += _replace_linears(child, factory, skip=skipped)
    return swapped


def _bnb_factory(mode: QuantMode) -> Any:
    import bitsandbytes as bnb

    def make(lin: nn.Linear) -> nn.Module:
        if mode == "nf4":
            q = bnb.nn.Linear4bit(
                lin.in_features,
                lin.out_features,
                bias=lin.bias is not None,
                quant_type="nf4",
                compute_dtype=torch.float32,
            )
        elif mode == "int8":
            q = bnb.nn.Linear8bitLt(
                lin.in_features,
                lin.out_features,
                bias=lin.bias is not None,
                has_fp16_weights=False,
            )
        else:
            raise QuantError(mode)
        with torch.no_grad():
            q.weight.data = lin.weight.data.clone()
            if lin.bias is not None and q.bias is not None:
                q.bias.data = lin.bias.data.clone()
        return q

    return make


@dataclass
class QuantReport:
    mode: str
    backend: str
    swapped: int
    bytes_before: int
    bytes_after: int
    ratio: float
    note: str


def _param_bytes(model: nn.Module) -> int:
    n = 0
    for p in model.parameters():
        n += p.numel() * p.element_size()
    for b in model.buffers():
        n += b.numel() * b.element_size()
    return n


class LinearQuantizer:
    def __init__(self, mode: QuantMode = "nf4", *, prefer_bnb: bool = True) -> None:
        if mode not in {"nf4", "int8", "none"}:
            raise QuantError(f"unknown mode {mode}")
        self.mode = mode
        self.prefer_bnb = prefer_bnb

    def apply(self, model: nn.Module, *, skip: Sequence[str] = ()) -> QuantReport:
        before = _param_bytes(model)
        if self.mode == "none":
            return QuantReport("none", "identity", 0, before, before, 1.0, "no-op")
        bnb = bitsandbytes_status()
        backend = "fallback"
        factory: Any
        note = ""
        if self.prefer_bnb and bnb["ok"]:
            try:
                factory = _bnb_factory(self.mode)
                backend = "bitsandbytes"
            except Exception as exc:
                LOG.warning("bnb wrap failed (%s); using fake quant", exc)
                factory = None
                note = str(exc)
        else:
            factory = None
            note = str(bnb.get("reason", "bnb disabled"))
        if factory is None:
            if self.mode == "nf4":
                factory = FakeLinear4Bit
            else:
                factory = FakeLinear8Bit
            backend = "fallback"
        swapped = _replace_linears(model, factory, skip=skip)
        if swapped == 0 and isinstance(model, nn.Linear):
            raise QuantError("root module is Linear; wrap it in nn.Sequential so we can setattr")
        after = _param_bytes(model)
        ratio = after / max(before, 1)
        return QuantReport(
            mode=self.mode,
            backend=backend,
            swapped=swapped,
            bytes_before=before,
            bytes_after=after,
            ratio=ratio,
            note=note,
        )


def _toy_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 16),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile 4/8-bit linear compression on a toy MLP.")
    p.add_argument("--mode", choices=("nf4", "int8", "none"), default="nf4")
    p.add_argument("--no-bnb", action="store_true")
    p.add_argument("--out", type=Path, default=Path("artifacts/quant.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model = _toy_mlp()
    x = torch.randn(8, 64)
    with torch.no_grad():
        y0 = model(x)
    report = LinearQuantizer(args.mode, prefer_bnb=not args.no_bnb).apply(model)
    with torch.no_grad():
        y1 = model(x)
    mae = float((y0 - y1).abs().mean())
    payload = {**asdict(report), "output_mae": mae, "bnb": bitsandbytes_status()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info(
        "%s via %s  swapped=%d  bytes %d -> %d  mae=%.4f",
        report.mode,
        report.backend,
        report.swapped,
        report.bytes_before,
        report.bytes_after,
        mae,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
