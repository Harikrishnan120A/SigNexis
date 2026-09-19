"""
dsp/fft_analysis.py
===================
FFT analysis using scipy.fft.

All frequency-axis values are mathematically correct:
    frequencies = np.fft.fftfreq(N, d=1/fs)  →  symmetric two-sided
    positive only = frequencies[:N//2]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import fft, fftfreq


@dataclass
class FFTResult:
    """Container for FFT analysis output."""

    freqs: np.ndarray         # positive-side frequency axis [Hz]
    magnitude: np.ndarray     # normalised magnitude spectrum (positive side)
    dominant_freq: float      # frequency of peak magnitude [Hz]
    spectral_centroid: float  # power-weighted mean frequency [Hz]
    spectral_bandwidth: float # weighted std around centroid [Hz]
    low_freq_energy: float    # fraction of energy below 250 Hz
    high_freq_energy: float   # fraction of energy above 2 kHz


def compute_fft(
    signal: np.ndarray,
    sampling_rate: float,
    low_cutoff: float = 250.0,
    high_cutoff: float = 2000.0,
) -> FFTResult:
    """Compute FFT and derived spectral features.

    Parameters
    ----------
    signal       : input time-domain signal
    sampling_rate: sample rate of *signal* in Hz
    low_cutoff   : boundary for 'low-frequency energy' in Hz
    high_cutoff  : boundary for 'high-frequency energy' in Hz

    Returns
    -------
    FFTResult
    """
    n = len(signal)
    if n == 0:
        raise ValueError("Cannot compute FFT of an empty signal.")

    # ── FFT ──────────────────────────────────────────────────────────────────
    spectrum = fft(signal)
    freqs_full = fftfreq(n, d=1.0 / sampling_rate)

    # Positive-frequency half only
    half = n // 2
    freqs = freqs_full[:half]
    magnitude = (2.0 / n) * np.abs(spectrum[:half])

    # DC bin: do not double it
    magnitude[0] /= 2.0

    # ── Dominant frequency ────────────────────────────────────────────────────
    peak_idx = int(np.argmax(magnitude))
    dominant_freq = float(freqs[peak_idx])

    # ── Spectral centroid and bandwidth ───────────────────────────────────────
    power = magnitude ** 2
    total_power = float(np.sum(power))

    if total_power > 0:
        spectral_centroid = float(np.sum(freqs * power) / total_power)
        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power)
        )
    else:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0

    # ── Band energies ─────────────────────────────────────────────────────────
    low_mask = freqs < low_cutoff
    high_mask = freqs >= high_cutoff

    low_freq_energy = (
        float(np.sum(power[low_mask]) / total_power) if total_power > 0 else 0.0
    )
    high_freq_energy = (
        float(np.sum(power[high_mask]) / total_power) if total_power > 0 else 0.0
    )

    return FFTResult(
        freqs=freqs,
        magnitude=magnitude,
        dominant_freq=dominant_freq,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        low_freq_energy=low_freq_energy,
        high_freq_energy=high_freq_energy,
    )
