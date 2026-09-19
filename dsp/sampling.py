"""
dsp/sampling.py
===============
Sampling, Nyquist analysis, and aliasing detection module.

Key relationships
-----------------
    Nyquist frequency  f_N = f_s / 2
    Properly sampled   iff  f_signal < f_N
    Undersampled       iff  f_signal >= f_N

    Apparent aliased frequency (when f_signal > f_N):
        f_alias = |f_signal - round(f_signal / f_s) * f_s|
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dsp.signal_generator import generate_reference_signal


@dataclass
class SamplingAnalysis:
    """Container for all Nyquist / aliasing results."""

    sampling_rate: float          # f_s   [Hz]
    nyquist_freq: float           # f_s / 2  [Hz]
    signal_freq: float            # true signal frequency  [Hz]
    is_properly_sampled: bool
    aliasing_detected: bool
    apparent_freq: Optional[float]  # aliased apparent frequency, None if N/A
    t_ref: np.ndarray             # high-res time axis
    signal_ref: np.ndarray        # high-res reference signal
    t_discrete: np.ndarray        # discrete time axis
    signal_discrete: np.ndarray   # sampled signal values
    status_message: str = field(default="")

    def __post_init__(self) -> None:
        if not self.status_message:
            self.status_message = self._build_status()

    def _build_status(self) -> str:
        if self.is_properly_sampled:
            return (
                f"✅ Properly sampled  "
                f"(f_signal={self.signal_freq:.1f} Hz  <  "
                f"f_Nyquist={self.nyquist_freq:.1f} Hz)"
            )
        else:
            msg = (
                f"⚠️ UNDERSAMPLED / Aliasing detected  "
                f"(f_signal={self.signal_freq:.1f} Hz  ≥  "
                f"f_Nyquist={self.nyquist_freq:.1f} Hz)"
            )
            if self.apparent_freq is not None:
                msg += f"\n    Apparent (aliased) frequency ≈ {self.apparent_freq:.1f} Hz"
            return msg


def analyse_sampling(
    signal_freq: float,
    sampling_rate: float,
    amplitude: float = 1.0,
    duration: float = 0.05,
) -> SamplingAnalysis:
    """Perform Nyquist analysis and return a :class:`SamplingAnalysis`.

    Parameters
    ----------
    signal_freq : float   – true signal frequency  [Hz]
    sampling_rate : float – discrete sampling rate  [Hz]
    amplitude : float     – signal amplitude
    duration : float      – analysis window length  [s]

    Returns
    -------
    SamplingAnalysis
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if signal_freq <= 0:
        raise ValueError("signal_freq must be positive.")

    nyquist_freq = sampling_rate / 2.0
    is_properly_sampled = signal_freq < nyquist_freq
    aliasing_detected = not is_properly_sampled

    # Apparent aliased frequency
    apparent_freq: Optional[float] = None
    if aliasing_detected:
        # Fold the frequency back into [0, f_s/2]
        # general formula: f_alias = |f_signal mod f_s|
        # if result > f_N, reflect: f_alias = f_s - result
        folded = signal_freq % sampling_rate
        apparent_freq = folded if folded <= nyquist_freq else sampling_rate - folded

    # Reference (high-resolution) signal
    t_ref, signal_ref = generate_reference_signal(
        signal_freq, amplitude, duration
    )

    # Discrete samples
    t_discrete = np.arange(0, duration, 1.0 / sampling_rate)
    signal_discrete = amplitude * np.sin(2 * np.pi * signal_freq * t_discrete)

    return SamplingAnalysis(
        sampling_rate=sampling_rate,
        nyquist_freq=nyquist_freq,
        signal_freq=signal_freq,
        is_properly_sampled=is_properly_sampled,
        aliasing_detected=aliasing_detected,
        apparent_freq=apparent_freq,
        t_ref=t_ref,
        signal_ref=signal_ref,
        t_discrete=t_discrete,
        signal_discrete=signal_discrete,
    )


def nyquist_frequency(sampling_rate: float) -> float:
    """Return the Nyquist frequency for a given sampling rate.

    f_N = f_s / 2
    """
    return sampling_rate / 2.0


def check_nyquist_condition(signal_freq: float, sampling_rate: float) -> bool:
    """Return True iff the Nyquist sampling theorem is satisfied."""
    return signal_freq < nyquist_frequency(sampling_rate)
