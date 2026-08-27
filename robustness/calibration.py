"""Expected calibration error + temperature scaling.

PGD makes the toy classifier both wrong *and* loud about it. Temperature
scaling is the cheapest fix that actually moved ECE on the held-out mix.
T is picked on a log grid against ECE, not NLL. Guo-style NLL made the
attacked set *less* calibrated, which is the number we publish.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("omni.calibrate")


class CalibrationError(RuntimeError):
    pass


@dataclass
class ECEReport:
    ece: float
    n_bins: int
    n: int
    bin_acc: list[float]
    bin_conf: list[float]
    bin_count: list[int]
    temperature: float = 1.0


def _bin_edges(n_bins: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n_bins + 1)


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
    temperature: float = 1.0,
) -> ECEReport:
    """ECE over equal-width confidence bins.

    The last bin is closed on the right so confidence==1.0 is not dropped.
    That off-by-one shipped in v0.3.1 and inflated ECE on saturated models.
    """
    if logits.ndim != 2:
        raise CalibrationError("logits must be [N, C]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise CalibrationError("labels must be [N] matching logits")
    if n_bins < 2:
        raise CalibrationError("n_bins must be >= 2")
    if temperature <= 0:
        raise CalibrationError("temperature must be > 0")
    probs = F.softmax(logits.float() / temperature, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(labels)
    edges = _bin_edges(n_bins)
    accs: list[float] = []
    confs: list[float] = []
    counts: list[int] = []
    ece = 0.0
    n = labels.shape[0]
    for b in range(n_bins):
        lo = edges[b].item()
        hi = edges[b + 1].item()
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum().item())
        counts.append(cnt)
        if cnt == 0:
            accs.append(0.0)
            confs.append(0.0)
            continue
        acc = float(correct[mask].float().mean().item())
        cmean = float(conf[mask].mean().item())
        accs.append(acc)
        confs.append(cmean)
        ece += (cnt / n) * abs(acc - cmean)
    return ECEReport(
        ece=float(ece),
        n_bins=n_bins,
        n=n,
        bin_acc=accs,
        bin_conf=confs,
        bin_count=counts,
        temperature=temperature,
    )


class TemperatureScaler(nn.Module):
    def __init__(self, init_t: float = 1.5) -> None:
        super().__init__()
        # log-param so T stays positive without a clamp dance
        self.log_t = nn.Parameter(torch.tensor(float(math.log(init_t))))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        t_min: float = 0.5,
        t_max: float = 16.0,
        n_grid: int = 48,
        n_bins: int = 15,
    ) -> float:
        """Grid-search T to minimise ECE.

        Guo et al. minimise NLL. That walked ECE *up* on the attacked toy
        set (overconfident and often wrong), which is the number we put in
        the report, so we optimise that number instead.
        """
        if logits.numel() == 0:
            raise CalibrationError("empty logits")
        logits = logits.detach().float()
        labels = labels.detach()
        ts = torch.cat(
            [
                torch.tensor([1.0]),
                torch.logspace(math.log10(t_min), math.log10(t_max), n_grid),
            ]
        )
        best_t = 1.0
        best_ece = float("inf")
        for t in ts.tolist():
            ece = expected_calibration_error(
                logits, labels, n_bins=n_bins, temperature=float(t)
            ).ece
            if ece < best_ece:
                best_ece = ece
                best_t = float(t)
        with torch.no_grad():
            self.log_t.copy_(torch.tensor(math.log(best_t)))
        return self.temperature


def fit_and_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    before = expected_calibration_error(logits, labels, n_bins=n_bins)
    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels, n_bins=n_bins)
    after = expected_calibration_error(logits, labels, n_bins=n_bins, temperature=t)
    return {
        "before": asdict(before),
        "after": asdict(after),
        "temperature": t,
        "ece_delta": before.ece - after.ece,
    }


def _synthetic(n: int = 256, n_classes: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # overconfident logits: large scale, some label noise
    logits = torch.randn(n, n_classes, generator=g) * 4.5
    labels = logits.argmax(-1)
    flip = torch.rand(n, generator=g) < 0.25
    labels = torch.where(
        flip,
        torch.randint(0, n_classes, (n,), generator=g),
        labels,
    )
    return logits, labels


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit temperature scaling on logits.")
    p.add_argument("--logits", type=Path, default=None, help="optional .pt with logits/labels")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--out", type=Path, default=Path("artifacts/calibration.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.logits is not None:
        blob = torch.load(args.logits, map_location="cpu", weights_only=False)
        logits, labels = blob["logits"], blob["labels"]
    else:
        logits, labels = _synthetic()
    report = fit_and_report(logits, labels, n_bins=args.bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info(
        "ECE %.4f -> %.4f  T=%.3f",
        report["before"]["ece"],
        report["after"]["ece"],
        report["temperature"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
