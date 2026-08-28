"""Aspect-ratio buckets for the image side of the mix.

Center-cropping everything to 224^2 wrecks UI screenshots and infographics,
which is a non-trivial slice of the web-image dump. We keep a short list of
canonical ratios, scale so the image *fits* the canvas, then pad. No stretch.

The batch sampler only mixes samples that landed in the same bucket so the
collate path does not have to pad twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as TF

LOG = logging.getLogger("omni.bucket")

Ratio = tuple[int, int]


def canonical_buckets() -> tuple[Ratio, ...]:
    # matches the internvl / qwen-vl short list more or less
    return (
        (1, 1),
        (4, 3),
        (3, 4),
        (16, 9),
        (9, 16),
        (3, 2),
        (2, 3),
    )


def _ratio_err(w: int, h: int, bucket: Ratio) -> float:
    if h == 0 or bucket[1] == 0:
        return float("inf")
    return abs(math.log((w / h) / (bucket[0] / bucket[1])))


def nearest_bucket(width: int, height: int, buckets: Sequence[Ratio] | None = None) -> Ratio:
    pool = tuple(buckets) if buckets is not None else canonical_buckets()
    return min(pool, key=lambda b: _ratio_err(width, height, b))


def canvas_size(bucket: Ratio, short_side: int) -> tuple[int, int]:
    a, b = bucket
    if a <= b:
        h = short_side
        w = int(round(short_side * a / b))
    else:
        w = short_side
        h = int(round(short_side * b / a))
    # keep both even; some conv stems still assume that
    w = max(2, w - (w % 2))
    h = max(2, h - (h % 2))
    return w, h


def pad_to_bucket(
    image: torch.Tensor,
    bucket: Ratio,
    short_side: int,
    fill: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Resize with aspect preserved, then pad to the bucket canvas.

    ``image`` is C,H,W in [0,1] or [0,255]; we do not care which, fill is
    in the same units.
    """
    if image.ndim != 3:
        raise ValueError(f"expected C,H,W got {tuple(image.shape)}")
    _, h, w = image.shape
    canvas_w, canvas_h = canvas_size(bucket, short_side)
    scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = TF.resize(image, [new_h, new_w], antialias=True)
    pad_w = canvas_w - new_w
    pad_h = canvas_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = TF.pad(resized, [left, top, right, bottom], fill=fill)
    meta = {
        "scale": float(scale),
        "pad_left": float(left),
        "pad_top": float(top),
        "pad_right": float(right),
        "pad_bottom": float(bottom),
        "src_w": float(w),
        "src_h": float(h),
        "canvas_w": float(canvas_w),
        "canvas_h": float(canvas_h),
    }
    return padded, meta


@dataclass
class ImageRecord:
    tensor: torch.Tensor
    caption: str
    bucket: Ratio
    path: str = ""


class AspectBucketDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        items: Sequence[tuple[torch.Tensor, str]],
        *,
        short_side: int = 224,
        buckets: Sequence[Ratio] | None = None,
        fill: float = 0.0,
        paths: Sequence[str] | None = None,
    ) -> None:
        self.short_side = short_side
        self.buckets = tuple(buckets) if buckets is not None else canonical_buckets()
        self.fill = fill
        self._items: list[ImageRecord] = []
        for i, (img, cap) in enumerate(items):
            if img.ndim != 3:
                raise ValueError(f"item {i}: expected C,H,W")
            _, h, w = img.shape
            bucket = nearest_bucket(int(w), int(h), self.buckets)
            path = "" if paths is None else paths[i]
            self._items.append(ImageRecord(tensor=img, caption=cap, bucket=bucket, path=path))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._items[idx]
        canvas, meta = pad_to_bucket(rec.tensor, rec.bucket, self.short_side, fill=self.fill)
        return {
            "image": canvas,
            "caption": rec.caption,
            "bucket": rec.bucket,
            "path": rec.path,
            "meta": meta,
        }

    def bucket_index(self) -> dict[Ratio, list[int]]:
        out: dict[Ratio, list[int]] = defaultdict(list)
        for i, rec in enumerate(self._items):
            out[rec.bucket].append(i)
        return dict(out)

    def histogram(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for rec in self._items:
            counts[f"{rec.bucket[0]}:{rec.bucket[1]}"] += 1
        return dict(sorted(counts.items()))


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: AspectBucketDataset,
        batch_size: int,
        *,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        batches: list[list[int]] = []
        # sort buckets so rank 0 and rank 1 walk the same order before shuffle
        index = self.dataset.bucket_index()
        for bucket in sorted(index, key=lambda b: (b[0], b[1])):
            order = list(index[bucket])
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _even_batches(self) -> list[list[int]]:
        """Same length on every rank. DDP hangs if one rank runs short.

        drop_last=True truncates. otherwise we repeat from the front, same
        as DistributedSampler, so the last step may duplicate a batch.
        """
        batches = self._global_batches()
        if self.num_replicas == 1:
            return batches
        if not batches:
            return []
        extra = len(batches) % self.num_replicas
        if extra == 0:
            return batches
        if self.drop_last:
            return batches[: len(batches) - extra]
        pad = self.num_replicas - extra
        out = list(batches)
        i = 0
        while pad:
            out.append(batches[i % len(batches)])
            i += 1
            pad -= 1
        return out

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._even_batches()
        yield from batches[self.rank :: self.num_replicas]

    def __len__(self) -> int:
        n = 0
        for idxs in self.dataset.bucket_index().values():
            if self.drop_last:
                n += len(idxs) // self.batch_size
            else:
                n += int(math.ceil(len(idxs) / self.batch_size))
        if self.num_replicas == 1:
            return n
        if n == 0:
            return 0
        if self.drop_last:
            n = n - (n % self.num_replicas)
            return n // self.num_replicas
        return int(math.ceil(n / self.num_replicas))


def collate_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty batch")
    buckets = {row["bucket"] for row in rows}
    if len(buckets) != 1:
        raise ValueError(f"mixed buckets in one batch: {buckets}")
    images = torch.stack([row["image"] for row in rows], dim=0)
    return {
        "image": images,
        "caption": [row["caption"] for row in rows],
        "bucket": rows[0]["bucket"],
        "path": [row["path"] for row in rows],
        "meta": [row["meta"] for row in rows],
    }


def make_loader(
    dataset: AspectBucketDataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    **sampler_kw: Any,
) -> DataLoader:
    sampler = BucketBatchSampler(dataset, batch_size, **sampler_kw)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_bucket,
    )


def _synthetic_items(n: int, seed: int = 0) -> list[tuple[torch.Tensor, str]]:
    rng = torch.Generator().manual_seed(seed)
    shapes = [(224, 224), (320, 180), (180, 320), (256, 192), (192, 256), (300, 200)]
    items: list[tuple[torch.Tensor, str]] = []
    for i in range(n):
        h, w = shapes[i % len(shapes)]
        img = torch.rand(3, h, w, generator=rng)
        items.append((img, f"synthetic caption {i}"))
    return items


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dump bucket occupancy for a synthetic mix.")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--short-side", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("artifacts/buckets.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = AspectBucketDataset(_synthetic_items(args.n), short_side=args.short_side)
    hist = ds.histogram()
    loader = make_loader(ds, args.batch_size, drop_last=False, shuffle=False)
    n_batches = 0
    shapes: dict[str, int] = {}
    for batch in loader:
        n_batches += 1
        key = f"{tuple(batch['image'].shape)}"
        shapes[key] = shapes.get(key, 0) + 1
    payload = {"histogram": hist, "batches": n_batches, "shapes": shapes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("buckets %s  batches=%d", hist, n_batches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
