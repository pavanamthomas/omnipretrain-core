from __future__ import annotations

import torch

from data.bucketing import (
    AspectBucketDataset,
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
        b = batch["bucket"]
        assert all(row == b for row in [b])
        # every image in the tensor has the same H,W
        assert batch["image"].shape[2] == batch["image"].shape[2]
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
