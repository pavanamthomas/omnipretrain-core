from __future__ import annotations

from pathlib import Path

from data.curriculum import main as curriculum_main
from training.checkpoints import main as ckpt_main
from robustness.calibration import main as cal_main
from optimization.quantizer import main as quant_main


def test_curriculum_cli(tmp_path: Path) -> None:
    inp = tmp_path / "caps.jsonl"
    inp.write_text('{"text":"A red bicycle leans against a wall."}\n{"text":"<div>x</div>"}\n')
    out = tmp_path / "out.jsonl"
    rc = curriculum_main(["--in", str(inp), "--out", str(out), "--max-epoch", "3", "--epoch", "2"])
    assert rc == 0
    assert out.exists()
    assert out.read_text().strip()


def test_ckpt_smoke(tmp_path: Path) -> None:
    rc = ckpt_main(["smoke", "--out", str(tmp_path), "--steps", "2"])
    assert rc == 0
    assert list(tmp_path.glob("step-*.pt"))


def test_calibrate_cli(tmp_path: Path) -> None:
    out = tmp_path / "cal.json"
    rc = cal_main(["--out", str(out), "--bins", "8"])
    assert rc == 0
    assert "ece" in out.read_text()


def test_quant_cli(tmp_path: Path) -> None:
    out = tmp_path / "q.json"
    rc = quant_main(["--mode", "nf4", "--no-bnb", "--out", str(out)])
    assert rc == 0
    assert "fallback" in out.read_text()
