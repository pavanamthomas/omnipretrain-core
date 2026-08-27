"""Background checkpoint writer.

Saving inside the training step was stalling rank0 for 2-4s on the NVMe box
once the state_dict grew past a couple hundred MB. The saver owns a bounded
queue and a single thread. Callers must pass CPU tensors; if you hand it a
CUDA storage the thread and the next step fight over the allocator.

Flush on close. The first version returned from ``submit`` and immediately
started the next step; the last checkpoint of a run never hit disk. That
was a fun 3am.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

LOG = logging.getLogger("omni.ckpt")


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointMeta:
    step: int
    path: Path
    nbytes: int
    write_ms: float
    extra: dict[str, Any]


class _Job:
    __slots__ = ("state", "step", "extra", "stop")

    def __init__(
        self,
        state: Mapping[str, torch.Tensor] | None,
        step: int,
        extra: dict[str, Any] | None,
        stop: bool = False,
    ) -> None:
        self.state = state
        self.step = step
        self.extra = extra or {}
        self.stop = stop


class AsyncCheckpointSaver:
    def __init__(
        self,
        out_dir: Path | str,
        *,
        max_inflight: int = 2,
        prefix: str = "step",
    ) -> None:
        if max_inflight < 1:
            raise CheckpointError("max_inflight must be >= 1")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._q: queue.Queue[_Job] = queue.Queue(maxsize=max_inflight)
        self._err: BaseException | None = None
        self._done: list[CheckpointMeta] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="ckpt-saver", daemon=True)
        self._closed = False
        self._thread.start()

    def submit(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        step: int,
        extra: dict[str, Any] | None = None,
        block: bool = True,
        timeout: float | None = 120.0,
    ) -> None:
        self._raise_if_dead()
        if self._closed:
            raise CheckpointError("saver is closed")
        cpu: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            t = v.detach()
            if t.device.type != "cpu":
                t = t.cpu()
            # clone so the caller can mutate the original storage
            cpu[k] = t.contiguous().clone()
        job = _Job(cpu, step, extra)
        try:
            self._q.put(job, block=block, timeout=timeout)
        except queue.Full as exc:
            raise CheckpointError("checkpoint queue is full; training would stall") from exc

    def flush(self, timeout: float = 300.0) -> list[CheckpointMeta]:
        self._raise_if_dead()
        self._q.join()
        with self._lock:
            return list(self._done)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(_Job(None, -1, None, stop=True))
        self._thread.join(timeout=60.0)
        self._raise_if_dead()

    def __enter__(self) -> "AsyncCheckpointSaver":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def written(self) -> list[CheckpointMeta]:
        with self._lock:
            return list(self._done)

    def _raise_if_dead(self) -> None:
        if self._err is not None:
            raise CheckpointError(f"background saver failed: {self._err}") from self._err

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job.stop:
                    return
                assert job.state is not None
                meta = _atomic_write(self.out_dir, self.prefix, job.step, job.state, job.extra)
                with self._lock:
                    self._done.append(meta)
                LOG.info("wrote %s (%d bytes, %.1f ms)", meta.path, meta.nbytes, meta.write_ms)
            except BaseException as exc:  # noqa: BLE001 — must not die silently
                self._err = exc
                LOG.exception("checkpoint write failed at step %s", job.step)
                return
            finally:
                self._q.task_done()


def _atomic_write(
    out_dir: Path,
    prefix: str,
    step: int,
    state: Mapping[str, torch.Tensor],
    extra: dict[str, Any],
) -> CheckpointMeta:
    final = out_dir / f"{prefix}-{step:06d}.pt"
    tmp = out_dir / f".{prefix}-{step:06d}.tmp"
    payload = {"step": step, "state": dict(state), "extra": extra}
    t0 = time.perf_counter()
    torch.save(payload, tmp)
    tmp.replace(final)
    nbytes = final.stat().st_size
    sidecar = final.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"step": step, "nbytes": nbytes, "extra": extra}, indent=2, default=str),
        encoding="utf-8",
    )
    return CheckpointMeta(
        step=step,
        path=final,
        nbytes=nbytes,
        write_ms=(time.perf_counter() - t0) * 1000,
        extra=extra,
    )


def inspect_ckpt(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("state", blob)
    keys = list(state) if isinstance(state, dict) else []
    return {
        "path": str(path),
        "step": blob.get("step"),
        "n_tensors": len(keys),
        "extra": blob.get("extra", {}),
        "keys_head": keys[:12],
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect or smoke-test async checkpoints.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="write a dummy tensor in the background")
    sm.add_argument("--out", type=Path, default=Path("artifacts/ckpts"))
    sm.add_argument("--steps", type=int, default=3)
    ins = sub.add_parser("inspect", help="print metadata for a .pt file")
    ins.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "inspect":
        print(json.dumps(inspect_ckpt(args.path), indent=2))
        return 0
    with AsyncCheckpointSaver(args.out) as saver:
        for step in range(1, args.steps + 1):
            saver.submit({"w": torch.randn(32, 32)}, step=step, extra={"note": "smoke"})
        written = saver.flush()
    LOG.info("wrote %d files", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
