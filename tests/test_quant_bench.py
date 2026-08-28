from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from optimization.benchmarker import GenerateWrapper, profile_model, render_table, upsert_markdown
from optimization.quantizer import LinearQuantizer, bitsandbytes_status


def test_fake_quant_swaps_and_runs() -> None:
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    x = torch.randn(4, 32)
    with torch.no_grad():
        y0 = model(x)
    report = LinearQuantizer("nf4", prefer_bnb=False).apply(model)
    assert report.swapped == 2
    assert report.backend == "fallback"
    assert report.bytes_after < report.bytes_before
    with torch.no_grad():
        y1 = model(x)
    assert y1.shape == y0.shape
    assert torch.isfinite(y1).all()


def test_int8_fallback() -> None:
    model = nn.Sequential(nn.Linear(16, 16))
    report = LinearQuantizer("int8", prefer_bnb=False).apply(model)
    assert report.swapped == 1
    y = model(torch.randn(2, 16))
    assert y.shape == (2, 16)


def test_bare_linear_is_rejected() -> None:
    from optimization.quantizer import QuantError

    try:
        LinearQuantizer("int8", prefer_bnb=False).apply(nn.Linear(8, 8))
    except QuantError:
        return
    raise AssertionError("expected QuantError")


def test_bnb_status_is_dict() -> None:
    st = bitsandbytes_status()
    assert "ok" in st
    assert "reason" in st


def test_benchmarker_writes_table(tmp_path: Path) -> None:
    model = GenerateWrapper(dim=32, vocab=64, depth=1, heads=4)
    rows = profile_model(
        model,
        batches=(1, 2),
        seq_len=8,
        new_tokens=2,
        warmup=1,
        iters=2,
        device="cpu",
    )
    assert len(rows) == 2
    assert rows[0].tok_s > 0
    md = tmp_path / "PERFORMANCE.md"
    md.write_text("# Performance\n\nintro\n", encoding="utf-8")
    upsert_markdown(md, render_table(rows))
    text = md.read_text(encoding="utf-8")
    assert "tok/s" in text
    assert "omnipretrain:perf-table:begin" in text
    upsert_markdown(md, render_table(rows))
    twice = md.read_text(encoding="utf-8")
    assert twice.count("omnipretrain:perf-table:begin") == 1
    assert twice.count("omnipretrain:perf-table:end") == 1
