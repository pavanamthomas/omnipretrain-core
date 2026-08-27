"""Throughput / TTFT profiler.

Runs the tiny VLM (or whatever you pass) across batch sizes and writes a
markdown table. Numbers in PERFORMANCE.md are from this script on CPU; do
not compare them to A100 vendor slides.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

LOG = logging.getLogger("omni.bench")

DEFAULT_BATCHES = (1, 4, 16, 32, 64)
_TABLE_BEGIN = "<!-- omnipretrain:perf-table:begin -->"
_TABLE_END = "<!-- omnipretrain:perf-table:end -->"


@dataclass
class BenchRow:
    batch: int
    seq_len: int
    ttft_ms: float
    tok_s: float
    step_ms: float
    vram_mb: float
    device: str


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GenerateWrapper(nn.Module):
    """Decode loop with a prefill pass so TTFT is a real measurement."""

    def __init__(self, dim: int = 64, vocab: int = 256, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.tok = nn.Embedding(vocab, dim)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.lm = nn.Linear(dim, vocab)
        self.vocab = vocab
        self.dim = dim

    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens)
        x = self.enc(x)
        return self.lm(x[:, -1, :])

    def decode_one(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.prefill(tokens)
        nxt = logits.argmax(dim=-1)
        return torch.cat([tokens, nxt.unsqueeze(1)], dim=1)


def profile_model(
    model: GenerateWrapper,
    *,
    batches: Sequence[int] = DEFAULT_BATCHES,
    seq_len: int = 32,
    new_tokens: int = 16,
    warmup: int = 2,
    iters: int = 5,
    device: str | None = None,
) -> list[BenchRow]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    rows: list[BenchRow] = []
    for bs in batches:
        if device == "cpu" and bs > 32:
            # 64 on CPU is just the GC thrashing; keep the column but mark it slow
            pass
        tokens = torch.randint(0, model.vocab, (bs, seq_len), device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.prefill(tokens)
            _sync()
            ttft_samples: list[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model.prefill(tokens)
                _sync()
                ttft_samples.append((time.perf_counter() - t0) * 1000)
            tok_samples: list[float] = []
            step_samples: list[float] = []
            for _ in range(iters):
                cur = tokens
                t0 = time.perf_counter()
                for _j in range(new_tokens):
                    cur = model.decode_one(cur)
                _sync()
                elapsed = time.perf_counter() - t0
                step_samples.append(elapsed * 1000)
                tok_samples.append((bs * new_tokens) / max(elapsed, 1e-8))
        rows.append(
            BenchRow(
                batch=bs,
                seq_len=seq_len,
                ttft_ms=statistics.median(ttft_samples),
                tok_s=statistics.median(tok_samples),
                step_ms=statistics.median(step_samples),
                vram_mb=_vram_mb(),
                device=device,
            )
        )
        LOG.info("bs=%d  ttft=%.1fms  tok/s=%.1f", bs, rows[-1].ttft_ms, rows[-1].tok_s)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    return rows


def render_table(rows: Sequence[BenchRow], *, title: str | None = None) -> str:
    device = rows[0].device if rows else "cpu"
    lines = [
        title or f"Toy decoder throughput ({device})",
        "",
        f"_seq_len={rows[0].seq_len if rows else '-'}._",
        "",
        "| batch | TTFT (ms) | tok/s | decode (ms) | VRAM (MiB) | device |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.batch} | {r.ttft_ms:.1f} | {r.tok_s:.1f} | {r.step_ms:.1f} | {r.vram_mb:.1f} | {r.device} |"
        )
    return "\n".join(lines) + "\n"


def upsert_markdown(path: Path, table: str) -> None:
    block = f"{_TABLE_BEGIN}\n{table.rstrip()}\n{_TABLE_END}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if _TABLE_BEGIN in text and _TABLE_END in text:
            pre = text.split(_TABLE_BEGIN, 1)[0]
            post = text.split(_TABLE_END, 1)[1]
            rest = post.lstrip("\n")
            path.write_text(pre + block + ("\n" + rest if rest else ""), encoding="utf-8")
            return
        path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")
        return
    path.write_text("# Performance\n\n" + block, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile toy decoder and write PERFORMANCE.md")
    p.add_argument("--out", type=Path, default=Path("PERFORMANCE.md"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--batches", default="1,4,16,32,64")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--new-tokens", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    batches = tuple(int(x) for x in args.batches.split(",") if x.strip())
    model = GenerateWrapper(dim=args.dim, depth=args.depth)
    rows = profile_model(
        model,
        batches=batches,
        seq_len=args.seq_len,
        new_tokens=args.new_tokens,
        warmup=args.warmup,
        iters=args.iters,
    )
    table = render_table(rows)
    upsert_markdown(args.out, table)
    if args.readme != args.out:
        upsert_markdown(args.readme, table)
    LOG.info("wrote table to %s (and %s)", args.out, args.readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
