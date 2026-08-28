from __future__ import annotations

import torch

from data.bucketing import (
    AspectBucketDataset,
    BucketBatchSampler,
    canonical_buckets,
    make_loader,
    nearest_bucket,
    pad_to_bucket,
)


def test_nearest_bucket_square() -> None:
    assert nearest_bucket(224, 224) == (1, 1)
    assert nearest_bucket(320, 180) == (16, 9)
    assert nearest_bucket(180, 320) == (9, 16)


def test_pad_does_not_stretch() -> None:
    img = torch.zeros(3, 100, 200)
    img[:, :, 0] = 1.0  # left column marker
    out, meta = pad_to_bucket(img, (1, 1), short_side=64)
    assert out.shape[1] == out.shape[2]
    # original aspect is 2:1 so we should have vertical padding
    assert meta["pad_top"] + meta["pad_bottom"] > 0
    assert meta["scale"] > 0
    # no stretch: after scale the content height/width ratio stays 1:2
    content_h = out.shape[1] - meta["pad_top"] - meta["pad_bottom"]
    content_w = out.shape[2] - meta["pad_left"] - meta["pad_right"]
    assert abs((content_w / content_h) - 2.0) < 0.08


def test_loader_never_mixes_buckets() -> None:
    items = []
    for i, (h, w) in enumerate([(64, 64), (64, 64), (48, 80), (48, 80), (80, 48)]):
        items.append((torch.rand(3, h, w), f"c{i}"))
    ds = AspectBucketDataset(items, short_side=32, buckets=canonical_buckets())
    loader = make_loader(ds, batch_size=2, drop_last=False, shuffle=False)
    seen = 0
    for batch in loader:
        seen += 1
        assert isinstance(batch["bucket"], tuple)
        assert len(batch["caption"]) == batch["image"].shape[0]
        h = batch["image"].shape[2]
        w = batch["image"].shape[3]
        for i in range(batch["image"].shape[0]):
            assert tuple(batch["image"][i].shape[-2:]) == (h, w)
    assert seen >= 2
    hist = ds.histogram()
    assert "1:1" in hist


def test_collate_rejects_mixed_ratios() -> None:
    from data.bucketing import collate_bucket

    a = {
        "image": torch.zeros(3, 8, 8),
        "caption": "a",
        "bucket": (1, 1),
        "path": "",
        "meta": {},
    }
    b = {
        "image": torch.zeros(3, 8, 16),
        "caption": "b",
        "bucket": (16, 9),
        "path": "",
        "meta": {},
    }
    try:
        collate_bucket([a, b])
    except ValueError as exc:
        assert "mixed" in str(exc).lower()
        return
    raise AssertionError("mixed buckets should raise")


def test_two_ranks_same_epoch_same_batch_count() -> None:
    items = [(torch.rand(3, 64, 64), f"c{i}") for i in range(16)]
    ds = AspectBucketDataset(items, short_side=32, buckets=((1, 1),))
    a = BucketBatchSampler(ds, 2, shuffle=True, seed=7, num_replicas=2, rank=0)
    b = BucketBatchSampler(ds, 2, shuffle=True, seed=7, num_replicas=2, rank=1)
    a.set_epoch(3)
    b.set_epoch(3)
    ba, bb = list(a), list(b)
    assert len(ba) == len(bb)
    assert len(ba) == len(a)
    assert len(bb) == len(b)


def test_two_ranks_drop_last_disjoint() -> None:
    items = [(torch.rand(3, 64, 64), f"c{i}") for i in range(20)]
    ds = AspectBucketDataset(items, short_side=32, buckets=((1, 1),))
    a = BucketBatchSampler(
        ds, 2, shuffle=True, seed=11, drop_last=True, num_replicas=2, rank=0
    )
    b = BucketBatchSampler(
        ds, 2, shuffle=True, seed=11, drop_last=True, num_replicas=2, rank=1
    )
    a.set_epoch(0)
    b.set_epoch(0)
    flat_a = [i for batch in a for i in batch]
    flat_b = [i for batch in b for i in batch]
    assert set(flat_a).isdisjoint(set(flat_b))
    # both ranks must still step; otherwise DDP waits forever
    assert flat_a and flat_b


def test_sampler_batches_are_one_bucket() -> None:
    items = [
        (torch.rand(3, h, w), f"c{i}")
        for i, (h, w) in enumerate(
            [(64, 64), (64, 64), (48, 80), (48, 80), (80, 48), (80, 48)]
        )
    ]
    ds = AspectBucketDataset(items, short_side=32, buckets=canonical_buckets())
    which: dict[int, tuple[int, int]] = {}
    for bucket, idxs in ds.bucket_index().items():
        for i in idxs:
            which[i] = bucket
    sampler = BucketBatchSampler(ds, 2, shuffle=True, seed=3)
    n = 0
    for batch in sampler:
        n += 1
        buckets = {which[i] for i in batch}
        assert len(buckets) == 1, buckets
    assert n >= 2
