"""
dsp/features.py
===============
Unified feature-extraction pipeline for SigNexis.

**This exact function must be used for both training and inference.**
Feature order is defined by FEATURE_NAMES and must match
models/feature_metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from dsp.fft_analysis import compute_fft

# ── Canonical feature order ────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    # Time-domain
    "mean",
    "std",
    "rms",
    "max_amplitude",
    "min_amplitude",
    "peak_to_peak",
    "zero_crossing_rate",
    # Frequency-domain
    "dominant_freq",
    "spectral_centroid",
    "spectral_bandwidth",
    "low_freq_energy",
    "high_freq_energy",
    # Signal quality
    "signal_power",
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(
    signal: np.ndarray,
    sampling_rate: float,
    clean_reference: np.ndarray | None = None,
) -> Dict[str, float]:
    """Extract the canonical feature set from a signal.

    Parameters
    ----------
    signal         : 1-D float array – the (possibly noisy) signal
    sampling_rate  : sample rate in Hz
    clean_reference: optional noiseless reference (same length as *signal*).
                     Used only for SNR – not included in the ML feature vector
                     because it is unavailable at inference time.

    Returns
    -------
    dict mapping each name in FEATURE_NAMES to a scalar float.
    NaN / Inf values are replaced with 0.0 to prevent model failures.
    """
    if len(signal) == 0:
        raise ValueError("Signal must not be empty.")

    # ── Time-domain ───────────────────────────────────────────────────────────
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    max_amp = float(np.max(signal))
    min_amp = float(np.min(signal))
    peak_to_peak = max_amp - min_amp

    n = len(signal)
    signs = np.sign(signal)
    signs = signs[signs != 0]
    sign_changes = np.sum(np.diff(signs) != 0) if len(signs) > 1 else 0
    zcr = float(sign_changes) / float(n - 1) if n > 1 else 0.0

    # ── Frequency-domain ─────────────────────────────────────────────────────
    try:
        fft_result = compute_fft(signal, sampling_rate)
        dominant_freq = fft_result.dominant_freq
        spectral_centroid = fft_result.spectral_centroid
        spectral_bandwidth = fft_result.spectral_bandwidth
        low_freq_energy = fft_result.low_freq_energy
        high_freq_energy = fft_result.high_freq_energy
    except Exception:
        dominant_freq = 0.0
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        low_freq_energy = 0.0
        high_freq_energy = 0.0

    # ── Power ─────────────────────────────────────────────────────────────────
    signal_power = float(np.mean(signal ** 2))

    # ── Assemble dict in canonical order ──────────────────────────────────────
    features: Dict[str, float] = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "max_amplitude": max_amp,
        "min_amplitude": min_amp,
        "peak_to_peak": peak_to_peak,
        "zero_crossing_rate": zcr,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "low_freq_energy": low_freq_energy,
        "high_freq_energy": high_freq_energy,
        "signal_power": signal_power,
    }

    # ── Sanitise ──────────────────────────────────────────────────────────────
    for key in features:
        v = features[key]
        if not np.isfinite(v):
            features[key] = 0.0

    return features


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a 1-D numpy array in canonical order."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def save_feature_metadata(path: Path | str = "models/feature_metadata.json") -> None:
    """Persist feature names and count so training / inference stay in sync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "num_features": NUM_FEATURES,
        "description": (
            "Canonical feature order for SigNexis RandomForest model. "
            "Both training and inference MUST use this exact order."
        ),
    }
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2)


def load_feature_metadata(path: Path | str = "models/feature_metadata.json") -> dict:
    """Load and return the feature metadata JSON."""
    with open(path) as fh:
        return json.load(fh)
