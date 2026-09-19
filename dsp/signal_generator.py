"""
dsp/signal_generator.py
=======================
Synthetic signal generator for SigNexis.

Supports six noise / distortion classes:
    CLEAN, GAUSSIAN_NOISE, LOW_FREQUENCY_NOISE, HIGH_FREQUENCY_NOISE,
    POWER_LINE_INTERFERENCE, DISTORTED_SIGNAL
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# ── public class names ────────────────────────────────────────────────────────
SIGNAL_CLASSES = [
    "CLEAN",
    "GAUSSIAN_NOISE",
    "LOW_FREQUENCY_NOISE",
    "HIGH_FREQUENCY_NOISE",
    "POWER_LINE_INTERFERENCE",
    "DISTORTED_SIGNAL",
]


def generate_signal(
    signal_type: str = "sine",
    frequency: float = 1000.0,
    amplitude: float = 1.0,
    duration: float = 1.0,
    sampling_rate: float = 8000.0,
    noise_type: str = "CLEAN",
    noise_amplitude: float = 0.3,
    *,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic signal with optional noise.

    Parameters
    ----------
    signal_type : {"sine", "multi_sine"}
        Base signal waveform.
    frequency : float
        Fundamental frequency in Hz.
    amplitude : float
        Peak amplitude of the clean signal.
    duration : float
        Signal duration in seconds.
    sampling_rate : float
        Discrete sampling rate in Hz (samples per second).
    noise_type : str
        One of SIGNAL_CLASSES.  "CLEAN" adds no noise.
    noise_amplitude : float
        Noise amplitude relative to signal amplitude.
    rng : np.random.Generator, optional
        Seeded random generator for reproducibility.

    Returns
    -------
    t_discrete : ndarray  –  discrete time axis [s]
    clean_signal : ndarray  –  noiseless signal at discrete samples
    noisy_signal : ndarray  –  signal after noise injection
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(0, duration, 1.0 / sampling_rate)

    # ── base waveform ─────────────────────────────────────────────────────────
    if signal_type == "multi_sine":
        # three harmonically related sines
        clean = (
            amplitude * np.sin(2 * np.pi * frequency * t)
            + 0.5 * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
            + 0.25 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
        )
    else:  # default: single sine
        clean = amplitude * np.sin(2 * np.pi * frequency * t)

    # ── noise injection ───────────────────────────────────────────────────────
    noisy = _add_noise(clean, t, noise_type, noise_amplitude, sampling_rate, rng)

    return t, clean, noisy


def _add_noise(
    clean: np.ndarray,
    t: np.ndarray,
    noise_type: str,
    noise_amp: float,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return clean + the requested noise."""
    n = len(clean)

    if noise_type == "CLEAN":
        return clean.copy()

    elif noise_type == "GAUSSIAN_NOISE":
        noise = rng.normal(0, noise_amp, n)
        return clean + noise

    elif noise_type == "LOW_FREQUENCY_NOISE":
        # low-frequency drift: 5 – 30 Hz tones
        lf_freq = rng.uniform(5, 30)
        noise = noise_amp * np.sin(2 * np.pi * lf_freq * t)
        return clean + noise

    elif noise_type == "HIGH_FREQUENCY_NOISE":
        # high-frequency interference: Nyquist/4 to Nyquist×0.9
        nyq = fs / 2.0
        hf_freq = rng.uniform(nyq * 0.5, nyq * 0.9)
        noise = noise_amp * np.sin(2 * np.pi * hf_freq * t)
        noise += rng.normal(0, noise_amp * 0.3, n)  # broadband HF component
        return clean + noise

    elif noise_type == "POWER_LINE_INTERFERENCE":
        # 50 Hz mains hum (+ 3rd harmonic)
        hum = noise_amp * np.sin(2 * np.pi * 50 * t)
        hum += 0.3 * noise_amp * np.sin(2 * np.pi * 150 * t)
        return clean + hum

    elif noise_type == "DISTORTED_SIGNAL":
        # hard-clipping + harmonic distortion
        distorted = clean.copy()
        clip_level = 0.7 * np.max(np.abs(clean)) if np.max(np.abs(clean)) > 0 else 0.7
        distorted = np.clip(distorted, -clip_level, clip_level)
        # add harmonic distortion
        distorted += 0.4 * noise_amp * np.sin(2 * np.pi * 3 * (fs / 8) * t)
        return distorted

    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. "
                         f"Choose from {SIGNAL_CLASSES}")


def generate_reference_signal(
    frequency: float,
    amplitude: float,
    duration: float,
    *,
    internal_fs: float = 192_000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """High-resolution 'continuous' reference signal for aliasing visualisation.

    Parameters
    ----------
    frequency : float
        Signal frequency in Hz.
    amplitude : float
        Peak amplitude.
    duration : float
        Duration in seconds.
    internal_fs : float
        Internal oversampling rate to approximate a continuous signal.

    Returns
    -------
    t_ref, signal_ref : ndarrays
    """
    t_ref = np.linspace(0, duration, int(internal_fs * duration), endpoint=False)
    signal_ref = amplitude * np.sin(2 * np.pi * frequency * t_ref)
    return t_ref, signal_ref
