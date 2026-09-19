"""
dsp/filters.py
==============
Digital filtering using scipy.signal.

Implements:
    low_pass_filter   – Butterworth LPF
    high_pass_filter  – Butterworth HPF
    band_pass_filter  – Butterworth BPF
    notch_filter      – IIR notch at a target frequency (default 50 Hz)

All cutoff frequencies are validated against the Nyquist frequency before
filter design.  Invalid parameters raise ValueError with a clear message.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt


def _zero_phase_filter(sos: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Use zero-phase filtering when padding is possible for the input length."""
    try:
        return sosfiltfilt(sos, signal)
    except ValueError as exc:
        if "padlen" not in str(exc):
            raise
        return sosfilt(sos, signal)


def _validate_cutoff(cutoff: float, nyquist: float, label: str = "cutoff") -> None:
    """Raise ValueError if cutoff is outside (0, nyquist)."""
    if cutoff <= 0:
        raise ValueError(f"{label} frequency must be positive, got {cutoff:.2f} Hz.")
    if cutoff >= nyquist:
        raise ValueError(
            f"{label} frequency ({cutoff:.2f} Hz) must be strictly less than "
            f"the Nyquist frequency ({nyquist:.2f} Hz)."
        )


def low_pass_filter(
    signal: np.ndarray,
    cutoff: float,
    sampling_rate: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal       : input 1-D signal
    cutoff       : cutoff frequency [Hz]
    sampling_rate: sampling rate [Hz]
    order        : filter order (default 5)

    Returns
    -------
    filtered : ndarray of same length as *signal*
    """
    nyquist = sampling_rate / 2.0
    _validate_cutoff(cutoff, nyquist, "LPF cutoff")
    sos = butter(order, cutoff / nyquist, btype="low", output="sos")
    return _zero_phase_filter(sos, signal)


def high_pass_filter(
    signal: np.ndarray,
    cutoff: float,
    sampling_rate: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    signal       : input 1-D signal
    cutoff       : cutoff frequency [Hz]
    sampling_rate: sampling rate [Hz]
    order        : filter order (default 5)

    Returns
    -------
    filtered : ndarray of same length as *signal*
    """
    nyquist = sampling_rate / 2.0
    _validate_cutoff(cutoff, nyquist, "HPF cutoff")
    sos = butter(order, cutoff / nyquist, btype="high", output="sos")
    return _zero_phase_filter(sos, signal)


def band_pass_filter(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sampling_rate: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    signal       : input 1-D signal
    low_cutoff   : lower cutoff frequency [Hz]
    high_cutoff  : upper cutoff frequency [Hz]
    sampling_rate: sampling rate [Hz]
    order        : filter order (default 5)

    Returns
    -------
    filtered : ndarray of same length as *signal*
    """
    nyquist = sampling_rate / 2.0
    _validate_cutoff(low_cutoff, nyquist, "BPF low_cutoff")
    _validate_cutoff(high_cutoff, nyquist, "BPF high_cutoff")
    if low_cutoff >= high_cutoff:
        raise ValueError(
            f"low_cutoff ({low_cutoff:.2f} Hz) must be less than "
            f"high_cutoff ({high_cutoff:.2f} Hz)."
        )
    Wn = [low_cutoff / nyquist, high_cutoff / nyquist]
    sos = butter(order, Wn, btype="band", output="sos")
    return _zero_phase_filter(sos, signal)


def notch_filter(
    signal: np.ndarray,
    notch_freq: float,
    sampling_rate: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply an IIR notch filter at *notch_freq* Hz.

    Default removes 50 Hz power-line interference.

    Parameters
    ----------
    signal        : input 1-D signal
    notch_freq    : frequency to attenuate [Hz]
    sampling_rate : sampling rate [Hz]
    quality_factor: Q factor – higher Q → narrower notch (default 30)

    Returns
    -------
    filtered : ndarray of same length as *signal*
    """
    nyquist = sampling_rate / 2.0
    _validate_cutoff(notch_freq, nyquist, "Notch frequency")
    # iirnotch uses normalised frequency w0 = notch_freq / (fs/2)
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return _zero_phase_filter(sos, signal)


# ── Filter recommendation ─────────────────────────────────────────────────────

FILTER_RECOMMENDATIONS = {
    "CLEAN": {
        "filter_type": "none",
        "reason": "Signal is classified as clean. No filtering required.",
        "params": {},
    },
    "GAUSSIAN_NOISE": {
        "filter_type": "low_pass",
        "reason": (
            "Gaussian (broadband) noise has significant high-frequency content. "
            "A low-pass filter attenuates high-frequency noise components."
        ),
        "params": {"cutoff_fraction": 0.4},  # fraction of Nyquist
    },
    "LOW_FREQUENCY_NOISE": {
        "filter_type": "high_pass",
        "reason": (
            "Low-frequency noise (baseline wander / drift) is below the signal band. "
            "A high-pass filter removes these unwanted low-frequency components."
        ),
        "params": {"cutoff_hz": 80.0},
    },
    "HIGH_FREQUENCY_NOISE": {
        "filter_type": "low_pass",
        "reason": (
            "High-frequency noise dominates above the signal band. "
            "A low-pass filter attenuates these high-frequency interference components."
        ),
        "params": {"cutoff_fraction": 0.3},  # fraction of Nyquist
    },
    "POWER_LINE_INTERFERENCE": {
        "filter_type": "notch",
        "reason": (
            "50 Hz mains (power-line) hum detected. "
            "A narrow notch filter removes the interference with minimal signal distortion."
        ),
        "params": {"notch_freq": 50.0, "quality_factor": 30.0},
    },
    "DISTORTED_SIGNAL": {
        "filter_type": "low_pass",
        "reason": (
            "Signal distortion introduces harmonic artefacts at high frequencies. "
            "A low-pass filter smooths the distorted waveform."
        ),
        "params": {"cutoff_fraction": 0.35},
    },
}


def apply_recommended_filter(
    signal: np.ndarray,
    sampling_rate: float,
    noise_class: str,
    signal_freq: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply the intelligent filter recommendation for a detected noise class.

    Parameters
    ----------
    signal        : noisy input signal
    sampling_rate : Hz
    noise_class   : one of the six SigNexis class names
    signal_freq   : fundamental signal frequency [Hz], used to set LPF cutoff
                    conservatively above signal_freq when available.

    Returns
    -------
    filtered : ndarray
    info     : dict with keys 'filter_type', 'reason', 'params', 'cutoff_used'
    """
    rec = FILTER_RECOMMENDATIONS.get(noise_class, FILTER_RECOMMENDATIONS["GAUSSIAN_NOISE"])
    nyquist = sampling_rate / 2.0
    info = dict(rec)
    info["cutoff_used"] = None

    if rec["filter_type"] == "none":
        return signal.copy(), info

    elif rec["filter_type"] == "low_pass":
        if signal_freq is not None and signal_freq > 0:
            # Set cutoff to 2× signal frequency but bounded by Nyquist
            cutoff = min(signal_freq * 2.5, nyquist * rec["params"]["cutoff_fraction"] * 2)
            cutoff = max(cutoff, signal_freq * 1.5)
            cutoff = min(cutoff, nyquist * 0.9)
        else:
            cutoff = nyquist * rec["params"]["cutoff_fraction"]
        cutoff = min(cutoff, nyquist * 0.9)
        info["cutoff_used"] = round(cutoff, 2)
        filtered = low_pass_filter(signal, cutoff, sampling_rate)
        return filtered, info

    elif rec["filter_type"] == "high_pass":
        cutoff = rec["params"]["cutoff_hz"]
        # Safety: don't exceed Nyquist
        cutoff = min(cutoff, nyquist * 0.9)
        info["cutoff_used"] = round(cutoff, 2)
        filtered = high_pass_filter(signal, cutoff, sampling_rate)
        return filtered, info

    elif rec["filter_type"] == "notch":
        notch_freq = rec["params"]["notch_freq"]
        q = rec["params"]["quality_factor"]
        if notch_freq >= nyquist:
            info["reason"] += (
                f"\n⚠️ Note: 50 Hz notch cannot be applied — "
                f"Nyquist = {nyquist:.1f} Hz."
            )
            return signal.copy(), info
        info["cutoff_used"] = notch_freq
        filtered = notch_filter(signal, notch_freq, sampling_rate, q)
        return filtered, info

    else:
        return signal.copy(), info
