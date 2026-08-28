"""Local DDP spawn. Kept off the default import path of TinyVLM unit tests
because init_process_group inside pytest poisons later cases — this file
goes through ``run_local`` which always spawn()s a fresh interpreter.
"""

from __future__ import annotations

from pathlib import Path

from training.distributed import TrainConfig, run_local, use_fsdp


def test_cpu_ddp_two_proc(tmp_path: Path) -> None:
    cfg = TrainConfig(
        world_size=2,
        steps=2,
        batch_size=2,
        seq_len=8,
        dim=32,
        depth=2,
        heads=4,
        image_size=32,
        patch=16,
        ckpt_dir=tmp_path / "ck",
        metrics_dir=tmp_path / "met",
        master_port=29917,
        save_every=2,
    )
    run_local(cfg)
    proms = list((tmp_path / "met").glob("rank*.prom"))
    jsonl = list((tmp_path / "met").glob("rank*.jsonl"))
    assert proms, "telemetry did not write a prom file"
    assert jsonl, "telemetry did not write jsonl"
    assert any(p.stat().st_size > 0 for p in proms)


def test_fsdp_needs_one_gpu_per_rank() -> None:
    assert use_fsdp(2, 2)
    assert use_fsdp(1, 1)
    assert not use_fsdp(2, 1)
    assert not use_fsdp(2, 0)
    assert not use_fsdp(1, 0)
