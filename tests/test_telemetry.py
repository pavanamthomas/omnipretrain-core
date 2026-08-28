from __future__ import annotations

from pathlib import Path

from training.telemetry import Telemetry, dump_summary


def test_prom_and_jsonl(tmp_path: Path) -> None:
    tel = Telemetry(rank=0, world_size=2, out_dir=tmp_path, wandb_project=None)
    tel.log_step(step=0, loss=1.2, grad_norm=0.4, step_s=0.05, tokens=128, vram_mb=0.0)
    tel.log_step(step=1, loss=1.1, grad_norm=0.3, step_s=0.04, tokens=128, vram_mb=0.0)
    tel.close()
    prom = (tmp_path / "rank0.prom").read_text(encoding="utf-8")
    assert "omni_train_loss" in prom
    assert 'rank="0"' in prom
    summary = dump_summary(tmp_path / "rank0.jsonl")
    assert summary["n"] == 2
    assert summary["loss_last"] == 1.1
    assert summary["tok_s_mean"] > 0
    assert tel.bundle.throughput() > 0


def test_jsonl_rotates_at_size_cap(tmp_path: Path) -> None:
    tel = Telemetry(
        rank=0,
        world_size=1,
        out_dir=tmp_path,
        wandb_project=None,
        max_jsonl_bytes=400,
        jsonl_keep=2,
    )
    for step in range(40):
        tel.log_step(step=step, loss=1.0, grad_norm=0.1, step_s=0.05, tokens=64, vram_mb=0.0)
    tel.close()
    rotated = tmp_path / "rank0.jsonl.1"
    assert rotated.exists(), "expected a rotated jsonl after the size cap"
    current = (tmp_path / "rank0.jsonl").read_text(encoding="utf-8")
    old = rotated.read_text(encoding="utf-8")
    assert current.strip() or old.strip()
    summary = dump_summary(tmp_path / "rank0.jsonl")
    assert summary["n"] >= 1
