"""Local FSDP simulation.

Real pretrain runs on 8xA100 with NCCL. This file is the laptop-shaped
version of that loop: spawn N processes, gloo backend, fully shard a tiny
VLM, run a few steps. Activation checkpointing is on by default because that
is the knob we actually care about validating (the wrap policy is easy to
get wrong and silently not wrap).

NCCL + spawn on a shared CPU box was a mess (timeouts, leftover processes).
gloo is slower and that is fine. torch 2.13 also refuses to FSDP-wrap on
CPU, so the no-GPU path is DDP; the CUDA path is still FSDP + wrap policy.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.checkpoint import checkpoint as ckpt_fn

from training.checkpoints import AsyncCheckpointSaver
from training.telemetry import Telemetry

LOG = logging.getLogger("omni.train")


class TrainError(RuntimeError):
    pass


@dataclass
class TrainConfig:
    world_size: int = 2
    steps: int = 8
    batch_size: int = 4
    seq_len: int = 32
    dim: int = 64
    depth: int = 4
    heads: int = 4
    vocab: int = 256
    patch: int = 16
    image_size: int = 64
    lr: float = 3e-4
    activation_ckpt: bool = True
    seed: int = 13
    backend: str = "gloo"
    ckpt_dir: Path = Path("artifacts/ckpts")
    save_every: int = 4
    metrics_dir: Path = Path("artifacts/metrics")
    master_port: int = 29731


class CheckpointBlock(nn.Module):
    def __init__(self, dim: int, heads: int, use_ckpt: bool) -> None:
        super().__init__()
        self.use_ckpt = use_ckpt
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ckpt and self.training:
            return ckpt_fn(self._inner, x, use_reentrant=False)
        return self._inner(x)


class TinyVLM(nn.Module):
    """Shared transformer over image patches + token embeddings.

    Not a real CLIP/LLaVA. It exists so FSDP wrap, grad-norm, and the
    checkpoint path have a graph that looks like the production one.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.dim)
        self.patch = nn.Conv2d(3, cfg.dim, kernel_size=cfg.patch, stride=cfg.patch)
        self.blocks = nn.ModuleList(
            [CheckpointBlock(cfg.dim, cfg.heads, cfg.activation_ckpt) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.lm = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.img_proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, _l = tokens.shape
        txt = self.tok(tokens)
        patches = self.patch(images)
        bsz, dim, ph, pw = patches.shape
        vis = patches.flatten(2).transpose(1, 2)
        vis = self.img_proj(vis)
        x = torch.cat([vis, txt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        txt_out = x[:, vis.size(1) :, :]
        return self.lm(txt_out)


def _init_dist(rank: int, world_size: int, backend: str, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    # gloo on CPU; if someone exports NCCL_DEBUG it is usually leftover from a cluster job
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def _fake_batch(cfg: TrainConfig, rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(cfg.seed + 1000 * rank + step)
    images = torch.rand(cfg.batch_size, 3, cfg.image_size, cfg.image_size, generator=g)
    tokens = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), generator=g)
    return images, tokens


def _grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm(2).item() ** 2)
    return total**0.5


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def _wrap_model(model: nn.Module, cfg: TrainConfig) -> nn.Module:
    """FSDP when there is a GPU. torch 2.13 refuses FSDP on CPU
    (``needs a non-CPU accelerator``); we DDP+gloo there so the spawn
    harness still runs on a laptop.
    """
    wrap = ModuleWrapPolicy({CheckpointBlock})
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        model = model.to(device_id)
        fsdp_kw: dict[str, Any] = {
            "auto_wrap_policy": wrap,
            "device_id": device_id,
        }
        return FSDP(model, **fsdp_kw)
    LOG.warning(
        "no accelerator; FSDP is GPU-only in this torch build, using DDP(gloo)"
    )
    return nn.parallel.DistributedDataParallel(model)


def _worker(rank: int, cfg: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank{rank} %(levelname)s %(name)s: %(message)s",
    )
    _init_dist(rank, cfg.world_size, cfg.backend, cfg.master_port)
    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % max(torch.cuda.device_count(), 1))
    model = TinyVLM(cfg)
    model = _wrap_model(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    telem = Telemetry(rank=rank, world_size=cfg.world_size, out_dir=cfg.metrics_dir)
    saver: AsyncCheckpointSaver | None = None
    if rank == 0:
        saver = AsyncCheckpointSaver(cfg.ckpt_dir)

    model.train()
    t_run = time.perf_counter()
    try:
        for step in range(cfg.steps):
            t0 = time.perf_counter()
            images, tokens = _fake_batch(cfg, rank, step)
            if torch.cuda.is_available():
                images = images.cuda()
                tokens = tokens.cuda()
            logits = model(images, tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab),
                tokens.reshape(-1),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = _grad_global_norm(model)
            opt.step()
            dt = time.perf_counter() - t0
            toks = cfg.batch_size * cfg.seq_len
            telem.log_step(
                step=step,
                loss=float(loss.detach()),
                grad_norm=gnorm,
                step_s=dt,
                tokens=toks,
                vram_mb=_vram_mb(),
            )
            if saver is not None and (step + 1) % cfg.save_every == 0:
                # materialise a cpu copy; do not hand FSDP handles to the thread
                cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                saver.submit(cpu_sd, step=step + 1, extra={"loss": float(loss.detach())})
            if rank == 0:
                LOG.info(
                    "step %d loss=%.4f gnorm=%.3f %.1f tok/s",
                    step,
                    float(loss.detach()),
                    gnorm,
                    toks / max(dt, 1e-6),
                )
        if saver is not None:
            saver.flush()
    finally:
        if saver is not None:
            saver.close()
        telem.close()
        torch.distributed.destroy_process_group()
        if rank == 0:
            LOG.info("done in %.1fs", time.perf_counter() - t_run)


def run_local(cfg: TrainConfig) -> None:
    if cfg.world_size < 1:
        raise TrainError("world_size must be >= 1")
    # spawn even for world_size=1 so the init/teardown path matches the
    # multi-proc job. in-process init_process_group in pytest is a trap.
    torch.multiprocessing.spawn(
        _worker,
        args=(cfg,),
        nprocs=cfg.world_size,
        join=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Spawn a local FSDP toy run.")
    p.add_argument("--world-size", "--world_size", dest="world_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--no-ckpt-act", action="store_true")
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/ckpts"))
    p.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    p.add_argument("--port", type=int, default=29731)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig(
        world_size=args.world_size,
        steps=args.steps,
        batch_size=args.batch_size,
        dim=args.dim,
        depth=args.depth,
        activation_ckpt=not args.no_ckpt_act,
        ckpt_dir=args.ckpt_dir,
        metrics_dir=args.metrics_dir,
        master_port=args.port,
    )
    run_local(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
