"""Hand-crafted vibration features: time-domain statistics, spectral band
energies, and envelope-spectrum amplitudes at the bearing's characteristic
fault frequencies.

Fault-frequency multipliers for the SKF 6205-2RS drive-end bearing, as published
by the CWRU Bearing Data Center (multiples of shaft speed):
    BPFI 5.4152, BPFO 3.5848, FTF 0.39828, BSF 4.7135
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

FAULT_ORDERS = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.39828}
N_BANDS = 16


def time_features(w: np.ndarray) -> dict[str, float]:
    x = w - w.mean()
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x)) + 1e-12
    return {
        "rms": rms,
        "std": x.std(),
        "peak": peak,
        "p2p": x.max() - x.min(),
        "kurtosis": kurtosis(x, fisher=False),
        "skewness": skew(x),
        "crest": peak / rms,
        "impulse": peak / mean_abs,
        "shape": rms / mean_abs,
        "clearance": peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12),
    }


def band_energies(w: np.ndarray, fs: int, n_bands: int = N_BANDS) -> dict[str, float]:
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    edges = np.linspace(0, len(spec), n_bands + 1, dtype=int)
    total = spec.sum() + 1e-12
    return {f"band{i:02d}": spec[a:b].sum() / total for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))}


def envelope_spectrum(w: np.ndarray, fs: int, band: tuple[float, float] = (2000.0, 5000.0)):
    """Band-pass, take the Hilbert envelope, return (freqs, amplitude spectrum)."""
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, w)))
    env -= env.mean()
    amp = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    freqs = np.fft.rfftfreq(len(env), 1 / fs)
    return freqs, amp


def fault_frequency_features(w: np.ndarray, fs: int, rpm: float, harmonics: int = 3) -> dict[str, float]:
    """Envelope amplitude at each characteristic frequency and its harmonics,
    normalised by the median envelope level so it is comparable across loads."""
    freqs, amp = envelope_spectrum(w, fs)
    fr = rpm / 60.0
    df = freqs[1] - freqs[0]
    floor = np.median(amp[(freqs > 10) & (freqs < 1000)]) + 1e-12
    out: dict[str, float] = {}
    for name, order in FAULT_ORDERS.items():
        for h in range(1, harmonics + 1):
            f0 = h * order * fr
            tol = max(2 * df, 0.02 * f0)  # small search band for slip
            sel = (freqs > f0 - tol) & (freqs < f0 + tol)
            out[f"env_{name}_h{h}"] = float(amp[sel].max() / floor) if sel.any() else 0.0
    return out


def extract(w: np.ndarray, fs: int, rpm: float) -> dict[str, float]:
    feats = time_features(w)
    feats.update(band_energies(w, fs))
    feats.update(fault_frequency_features(w, fs, rpm))
    return feats


FEATURE_GROUPS = {
    "time": lambda c: not c.startswith(("band", "env_")),
    "spectral": lambda c: c.startswith("band"),
    "envelope": lambda c: c.startswith("env_"),
}
