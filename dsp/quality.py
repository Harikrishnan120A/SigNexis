"""
dsp/quality.py
==============
Signal-quality metrics for SigNexis.

Computes RMS, peak amplitude, dominant frequency, signal power, noise power,
and SNR.

**Important engineering note on SNR:**
    SNR = 10 · log10(P_signal / P_noise)

    Absolute SNR requires a *known clean reference*.
    When no reference is available, this module clearly labels the result
    as an *estimate* based on the dominant-frequency component, not a
    guaranteed ground-truth value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dsp.fft_analysis import compute_fft


@dataclass
class QualityMetrics:
    """Signal quality metrics container."""

    rms: float
    peak_amplitude: float
    dominant_freq: float
    signal_power: float
    noise_power: Optional[float]      # None when reference unavailable
    snr_db: Optional[float]           # None when reference unavailable
    snr_note: str                     # explanation of SNR method used


def compute_quality(
    signal: np.ndarray,
    sampling_rate: float,
    clean_reference: Optional[np.ndarray] = None,
) -> QualityMetrics:
    """Compute quality metrics for *signal*.

    Parameters
    ----------
    signal         : 1-D float array
    sampling_rate  : Hz
    clean_reference: optional noiseless signal of the same length.
                     When provided, true SNR is computed.

    Returns
    -------
    QualityMetrics
    """
    if len(signal) == 0:
        raise ValueError("Signal must not be empty.")

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak_amplitude = float(np.max(np.abs(signal)))
    signal_power = float(np.mean(signal ** 2))

    fft_res = compute_fft(signal, sampling_rate)
    dominant_freq = fft_res.dominant_freq

    # ── SNR calculation ───────────────────────────────────────────────────────
    if clean_reference is not None and len(clean_reference) == len(signal):
        # True SNR: clean reference available
        noise_component = signal - clean_reference
        noise_power = float(np.mean(noise_component ** 2))
        ref_power = float(np.mean(clean_reference ** 2))

        if noise_power > 0 and ref_power > 0:
            snr_db = float(10.0 * np.log10(ref_power / noise_power))
        elif noise_power == 0:
            snr_db = float("inf")
        else:
            snr_db = None

        snr_note = (
            "SNR computed from clean reference: "
            "SNR = 10·log₁₀(P_signal / P_noise)"
        )
    else:
        # No reference – estimate from spectral decomposition
        noise_power = None
        snr_db = None
        snr_note = (
            "⚠️  No clean reference available. "
            "Absolute SNR cannot be reliably calculated from the noisy signal alone. "
            "SNR improvement is shown as the change in estimated RMS-based power ratio."
        )

    return QualityMetrics(
        rms=rms,
        peak_amplitude=peak_amplitude,
        dominant_freq=dominant_freq,
        signal_power=signal_power,
        noise_power=noise_power,
        snr_db=snr_db,
        snr_note=snr_note,
    )


def snr_improvement(
    before: QualityMetrics,
    after: QualityMetrics,
) -> Optional[float]:
    """Return SNR improvement in dB when both SNR values are available."""
    if before.snr_db is not None and after.snr_db is not None:
        return after.snr_db - before.snr_db
    return None


def estimate_snr_rms_ratio(
    noisy: np.ndarray,
    filtered: np.ndarray,
) -> float:
    """Estimate SNR improvement from the RMS reduction achieved by filtering.

    This is a heuristic:  improvement ≈ 20·log₁₀(RMS_noisy / RMS_filtered)
    when the filter mainly removes noise and the signal is preserved.
    Always label this as an estimate in the UI.
    """
    rms_before = float(np.sqrt(np.mean(noisy ** 2)))
    rms_after = float(np.sqrt(np.mean(filtered ** 2)))
    if rms_after > 0 and rms_before > 0:
        return float(20.0 * np.log10(rms_before / rms_after))
    return 0.0
