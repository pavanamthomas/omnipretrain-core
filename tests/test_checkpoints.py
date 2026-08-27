from __future__ import annotations

from pathlib import Path

import torch

from training.checkpoints import AsyncCheckpointSaver, inspect_ckpt


def test_async_write_and_flush(tmp_path: Path) -> None:
    saver = AsyncCheckpointSaver(tmp_path, max_inflight=2)
    try:
        for step in range(1, 4):
            saver.submit({"w": torch.ones(8, 8) * step}, step=step, extra={"k": step})
        written = saver.flush()
    finally:
        saver.close()
    assert len(written) == 3
    paths = list(tmp_path.glob("step-*.pt"))
    assert len(paths) == 3
    meta = inspect_ckpt(tmp_path / "step-000003.pt")
    assert meta["step"] == 3
    assert meta["n_tensors"] == 1
    blob = torch.load(tmp_path / "step-000002.pt", map_location="cpu", weights_only=False)
    assert torch.equal(blob["state"]["w"], torch.ones(8, 8) * 2)


def test_closed_submit_raises(tmp_path: Path) -> None:
    saver = AsyncCheckpointSaver(tmp_path)
    saver.close()
    try:
        saver.submit({"w": torch.zeros(2)}, step=1)
    except Exception:
        return
    raise AssertionError("submit after close should fail")
