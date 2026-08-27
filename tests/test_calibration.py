from __future__ import annotations

import torch

from robustness.calibration import TemperatureScaler, expected_calibration_error, fit_and_report


def test_perfect_calibration_is_zero() -> None:
    # large margin so softmax saturates; last bin must still count conf==1.0
    logits = torch.tensor([[40.0, 0.0], [0.0, 40.0], [40.0, 0.0], [0.0, 40.0]])
    labels = torch.tensor([0, 1, 0, 1])
    report = expected_calibration_error(logits, labels, n_bins=10)
    assert report.ece < 1e-4
    assert report.bin_count[-1] == 4  # all mass in the last bin (the v0.3.1 bug)


def test_temperature_reduces_ece_when_overconfident() -> None:
    torch.manual_seed(0)
    n, c = 400, 6
    logits = torch.randn(n, c) * 6.0
    labels = logits.argmax(-1)
    flip = torch.rand(n) < 0.3
    rnd = torch.randint(0, c, (n,))
    labels = torch.where(flip, rnd, labels)
    out = fit_and_report(logits, labels, n_bins=15)
    # not a theorem, but with this synthetic it has been reliably true
    assert out["after"]["ece"] <= out["before"]["ece"] + 0.02
    assert out["temperature"] > 0


def test_scaler_positive_t() -> None:
    logits = torch.randn(32, 3)
    labels = torch.randint(0, 3, (32,))
    t = TemperatureScaler(init_t=1.2).fit(logits, labels)
    assert t > 0
