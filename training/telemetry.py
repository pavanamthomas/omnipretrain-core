"""Step metrics -> Prometheus textfile + optional W&B.

Cluster jobs already scrape node_exporter textfiles off the shared FS, so the
primary sink is a .prom file, not an HTTP server. W&B is best-effort; if the
key is missing we just skip it. I do not want training to die because a
dashboard is down.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

LOG = logging.getLogger("omni.telem")


@dataclass
class StepRow:
    step: int
    rank: int
    loss: float
    grad_norm: float
    step_s: float
    tokens_per_s: float
    vram_mb: float
    wall_s: float


@dataclass
class MetricsBundle:
    rows: list[StepRow] = field(default_factory=list)

    def throughput(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.tokens_per_s for r in self.rows) / len(self.rows)

    def last(self) -> StepRow | None:
        return self.rows[-1] if self.rows else None


def _prom_registry(prefix: str) -> tuple[CollectorRegistry, dict[str, Gauge]]:
    reg = CollectorRegistry()
    labels = ("rank", "world")
    gauges = {
        "loss": Gauge(f"{prefix}_loss", "train loss", labels, registry=reg),
        "grad_norm": Gauge(f"{prefix}_grad_norm", "global grad L2", labels, registry=reg),
        "step_seconds": Gauge(f"{prefix}_step_seconds", "step wall time", labels, registry=reg),
        "tokens_per_second": Gauge(
            f"{prefix}_tokens_per_second", "tokens / step time", labels, registry=reg
        ),
        "vram_mb": Gauge(f"{prefix}_vram_mb", "peak cuda MiB", labels, registry=reg),
        "step": Gauge(f"{prefix}_step", "global step", labels, registry=reg),
    }
    return reg, gauges


class Telemetry:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        out_dir: Path | str,
        prefix: str = "omni_train",
        wandb_project: str | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.bundle = MetricsBundle()
        self._t0 = time.perf_counter()
        self._prom_path = self.out_dir / f"rank{rank}.prom"
        self._jsonl_path = self.out_dir / f"rank{rank}.jsonl"
        self._jsonl: TextIO = self._jsonl_path.open("a", encoding="utf-8")
        self._reg, self._gauges = _prom_registry(prefix)
        self._wandb: Any = None
        project = wandb_project or os.environ.get("WANDB_PROJECT")
        mode = os.environ.get("WANDB_MODE", "")
        if project and mode != "disabled" and rank == 0:
            self._wandb = _try_wandb(project)

    def log_step(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: float,
        step_s: float,
        tokens: int,
        vram_mb: float,
    ) -> StepRow:
        row = StepRow(
            step=step,
            rank=self.rank,
            loss=loss,
            grad_norm=grad_norm,
            step_s=step_s,
            tokens_per_s=tokens / max(step_s, 1e-8),
            vram_mb=vram_mb,
            wall_s=time.perf_counter() - self._t0,
        )
        self.bundle.rows.append(row)
        self._jsonl.write(json.dumps(asdict(row)) + "\n")
        self._jsonl.flush()
        lab = {"rank": str(self.rank), "world": str(self.world_size)}
        self._gauges["loss"].labels(**lab).set(row.loss)
        self._gauges["grad_norm"].labels(**lab).set(row.grad_norm)
        self._gauges["step_seconds"].labels(**lab).set(row.step_s)
        self._gauges["tokens_per_second"].labels(**lab).set(row.tokens_per_s)
        self._gauges["vram_mb"].labels(**lab).set(row.vram_mb)
        self._gauges["step"].labels(**lab).set(row.step)
        # node_exporter textfile collector wants a rename-into-place; the helper does that
        write_to_textfile(str(self._prom_path), self._reg)
        if self._wandb is not None:
            try:
                self._wandb.log(
                    {
                        "train/loss": row.loss,
                        "train/grad_norm": row.grad_norm,
                        "train/tok_s": row.tokens_per_s,
                        "train/vram_mb": row.vram_mb,
                    },
                    step=step,
                )
            except Exception as exc:  # wandb outages should not kill the run
                LOG.warning("wandb.log failed: %s", exc)
        return row

    def close(self) -> None:
        self._jsonl.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception as exc:
                LOG.warning("wandb.finish failed: %s", exc)


def _try_wandb(project: str) -> Any | None:
    try:
        import wandb
    except ImportError:
        LOG.info("wandb not installed; skipping")
        return None
    key = os.environ.get("WANDB_API_KEY")
    if not key and os.environ.get("WANDB_MODE") != "offline":
        LOG.info("no WANDB_API_KEY; skipping wandb")
        return None
    try:
        return wandb.init(project=project, mode=os.environ.get("WANDB_MODE", "online"))
    except Exception as exc:
        LOG.warning("wandb.init failed: %s", exc)
        return None


def dump_summary(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return {"n": 0}
    losses = [r["loss"] for r in rows]
    return {
        "n": len(rows),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "tok_s_mean": sum(r["tokens_per_s"] for r in rows) / len(rows),
        "grad_norm_last": rows[-1]["grad_norm"],
        "vram_mb_max": max(r["vram_mb"] for r in rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarise a telemetry jsonl.")
    p.add_argument("path", type=Path)
    args = p.parse_args(argv)
    print(json.dumps(dump_summary(args.path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
