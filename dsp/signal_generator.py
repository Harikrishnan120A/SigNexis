from __future__ import annotations
import numpy as np

SIGNAL_CLASSES = ["CLEAN", "GAUSSIAN_NOISE", "LOW_FREQUENCY_NOISE", "HIGH_FREQUENCY_NOISE", "POWER_LINE_INTERFERENCE", "DISTORTED_SIGNAL"]

def generate_signal(signal_type="sine", frequency=1000.0, amplitude=1.0, duration=1.0, sampling_rate=8000.0, noise_type="CLEAN", noise_amplitude=0.3, *, rng=None):
    rng = rng or np.random.default_rng()
    t = np.arange(0, duration, 1.0 / sampling_rate)
    if signal_type == "multi_sine":
        clean = amplitude * (np.sin(2*np.pi*frequency*t) + .5*np.sin(4*np.pi*frequency*t) + .25*np.sin(6*np.pi*frequency*t))
    else:
        clean = amplitude * np.sin(2*np.pi*frequency*t)
    return t, clean, _add_noise(clean, t, noise_type, noise_amplitude, sampling_rate, rng)

def _add_noise(clean, t, noise_type, noise_amp, fs, rng):
    n = len(clean)
    if noise_type == "CLEAN": return clean.copy()
    if noise_type == "GAUSSIAN_NOISE": return clean + rng.normal(0, noise_amp, n)
    if noise_type == "LOW_FREQUENCY_NOISE": return clean + noise_amp*np.sin(2*np.pi*rng.uniform(5, 30)*t)
    if noise_type == "HIGH_FREQUENCY_NOISE":
        noise = noise_amp*np.sin(2*np.pi*rng.uniform(fs*.25, fs*.45)*t)
        return clean + noise + rng.normal(0, noise_amp*.3, n)
    if noise_type == "POWER_LINE_INTERFERENCE": return clean + noise_amp*np.sin(2*np.pi*50*t) + .3*noise_amp*np.sin(2*np.pi*150*t)
    if noise_type == "DISTORTED_SIGNAL":
        level = .7*np.max(np.abs(clean)) if np.max(np.abs(clean)) else .7
        return np.clip(clean, -level, level) + .4*noise_amp*np.sin(2*np.pi*3*(fs/8)*t)
    raise ValueError(f"Unknown noise_type '{noise_type}'. Choose from {SIGNAL_CLASSES}")

def generate_reference_signal(frequency, amplitude, duration, *, internal_fs=192000.0):
    t = np.linspace(0, duration, int(internal_fs*duration), endpoint=False)
    return t, amplitude*np.sin(2*np.pi*frequency*t)
