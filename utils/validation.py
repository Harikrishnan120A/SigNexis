"""
utils/validation.py
===================
Input validation utilities for SigNexis.
"""

from __future__ import annotations

from typing import Tuple


def validate_sampling_rate(fs: float) -> Tuple[bool, str]:
    """Return (ok, message)."""
    if fs <= 0:
        return False, f"Sampling rate must be positive (got {fs})."
    if fs < 100:
        return False, f"Sampling rate {fs} Hz is very low. Minimum recommended: 100 Hz."
    if fs > 192_000:
        return False, f"Sampling rate {fs} Hz exceeds maximum supported (192 000 Hz)."
    return True, "OK"


def validate_signal_frequency(f_sig: float, fs: float) -> Tuple[bool, str]:
    """Return (ok, message). Does NOT reject aliased frequencies – only validates input."""
    if f_sig <= 0:
        return False, f"Signal frequency must be positive (got {f_sig})."
    if f_sig >= fs / 2:
        msg = (
            f"⚠️  f_signal ({f_sig:.1f} Hz) ≥ Nyquist ({fs/2:.1f} Hz). "
            "Aliasing will occur."
        )
        return True, msg   # valid input but aliasing will be shown
    return True, "OK"


def validate_cutoff_frequency(
    cutoff: float,
    fs: float,
    label: str = "Cutoff",
) -> Tuple[bool, str]:
    """Return (ok, message)."""
    nyquist = fs / 2.0
    if cutoff <= 0:
        return False, f"{label} must be positive (got {cutoff})."
    if cutoff >= nyquist:
        return False, (
            f"{label} ({cutoff:.1f} Hz) must be strictly less than "
            f"Nyquist ({nyquist:.1f} Hz)."
        )
    return True, "OK"


def validate_duration(duration: float) -> Tuple[bool, str]:
    """Return (ok, message)."""
    if duration <= 0:
        return False, "Duration must be positive."
    if duration > 60:
        return False, "Duration exceeds 60 s limit for synthetic signals."
    return True, "OK"
