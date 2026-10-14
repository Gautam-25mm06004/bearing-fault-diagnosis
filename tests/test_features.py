"""Unit tests that need no dataset: run with `python -m pytest -q`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bearing.data import windows  # noqa: E402
from bearing.evaluate import location_of  # noqa: E402
from bearing.features import FAULT_ORDERS, extract, fault_frequency_features, time_features  # noqa: E402

FS = 12_000


def test_windows_shape_and_overlap():
    x = np.arange(10_000, dtype=float)
    w = windows(x, size=2048, step=1024)
    assert w.shape == (8, 2048)
    assert w[1, 0] == 1024  # hop respected


def test_time_features_sine():
    t = np.arange(4096) / FS
    f = time_features(np.sin(2 * np.pi * 50 * t))
    assert abs(f["rms"] - 1 / np.sqrt(2)) < 1e-2
    assert abs(f["crest"] - np.sqrt(2)) < 1e-2
    assert abs(f["kurtosis"] - 1.5) < 0.05  # kurtosis of a sine


def test_envelope_detects_simulated_inner_race_fault():
    """Impulses repeating at BPFI, ringing at 3.5 kHz, must light up the BPFI feature."""
    rng = np.random.default_rng(0)
    rpm = 1797.0
    bpfi = FAULT_ORDERS["BPFI"] * rpm / 60
    n = 8192
    t = np.arange(n) / FS
    x = 0.05 * rng.standard_normal(n)
    ring = np.exp(-np.arange(200) / 25) * np.sin(2 * np.pi * 3500 * np.arange(200) / FS)
    for k in np.arange(0, t[-1], 1 / bpfi):
        i = int(k * FS)
        x[i:i + 200] += ring[: max(0, min(200, n - i))]
    f = fault_frequency_features(x, FS, rpm)
    assert f["env_BPFI_h1"] > 3 * f["env_BPFO_h1"]


def test_extract_has_all_groups():
    f = extract(np.random.default_rng(1).standard_normal(2048), FS, 1797.0)
    assert any(k.startswith("band") for k in f)
    assert any(k.startswith("env_") for k in f)
    assert np.isfinite(list(f.values())).all()


def test_location_mapping():
    s = pd.Series(["Normal", "B014", "IR021", "OR007@6"]).map(location_of)
    assert list(s) == ["Normal", "Ball", "InnerRace", "OuterRace"]
