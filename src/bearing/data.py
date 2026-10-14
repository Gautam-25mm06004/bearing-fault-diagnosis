"""Loading the CWRU 12 kHz drive-end bearing data and cutting it into windows.

Dataset: Case Western Reserve University Bearing Data Center (12 kHz drive-end
accelerometer, SKF 6205-2RS bearing, motor loads 0-3 hp). The .mat files are
fetched by ``scripts/download_data.py`` from the public GitHub mirror
XiongMeijing/CWRU-1, which keeps the original CWRU file contents.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

FS = 12_000  # samples per second for every file used here

# The standard 10-class CWRU task: normal plus three fault sizes at three locations.
# Outer-race faults use the centred (@6 o'clock) position, as in most papers.
CLASSES = [
    "Normal",
    "B007", "B014", "B021",
    "IR007", "IR014", "IR021",
    "OR007@6", "OR014@6", "OR021@6",
]
LOADS = [0, 1, 2, 3]  # motor load in horsepower
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


@dataclass
class Recording:
    label: str
    load: int
    signal: np.ndarray  # drive-end acceleration, 1-D float64
    rpm: float
    source: str


def file_for(data_dir: Path, label: str, load: int) -> Path:
    if label == "Normal":
        return data_dir / "Normal" / f"Normal_{load}.mat"
    return data_dir / "12k_DE" / f"{label}_{load}.mat"


def load_recording(data_dir: Path, label: str, load: int) -> Recording:
    path = file_for(Path(data_dir), label, load)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    mat = loadmat(path)
    de_keys = [k for k in mat if re.search(r"_DE_time$", k)]
    if not de_keys:
        raise KeyError(f"No drive-end channel in {path.name}: {list(mat)}")
    # Some files carry a second recording; the one whose number matches the
    # RPM key is the intended one. Fall back to the first DE channel.
    rpm_keys = [k for k in mat if k.endswith("RPM")]
    key = de_keys[0]
    rpm = float(NOMINAL_RPM[load])
    if rpm_keys:
        prefix = rpm_keys[0].replace("RPM", "")
        match = [k for k in de_keys if k.startswith(prefix)]
        key = match[0] if match else key
        rpm = float(np.ravel(mat[rpm_keys[0]])[0])
    signal = np.asarray(mat[key], dtype=np.float64).ravel()
    return Recording(label=label, load=load, signal=signal, rpm=rpm, source=path.name)


def load_all(data_dir: Path) -> list[Recording]:
    return [load_recording(data_dir, c, l) for c in CLASSES for l in LOADS]


def windows(signal: np.ndarray, size: int = 2048, step: int | None = None) -> np.ndarray:
    """Cut a 1-D signal into windows of ``size`` samples.

    ``step`` defaults to ``size`` (no overlap). Overlapping windows from the same
    recording are highly correlated, which is exactly why a random window split
    leaks information between train and test.
    """
    step = step or size
    n = 1 + (len(signal) - size) // step
    if n <= 0:
        return np.empty((0, size))
    idx = np.arange(size)[None, :] + step * np.arange(n)[:, None]
    return signal[idx]
