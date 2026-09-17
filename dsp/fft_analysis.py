from dataclasses import dataclass
import numpy as np
from scipy.fft import fft, fftfreq

@dataclass
class FFTResult:
    freqs: np.ndarray
    magnitude: np.ndarray
    dominant_freq: float
    spectral_centroid: float
    spectral_bandwidth: float
    low_freq_energy: float
    high_freq_energy: float

def compute_fft(signal, sampling_rate, low_cutoff=250.0, high_cutoff=2000.0):
    n = len(signal)
    if n == 0: raise ValueError("Cannot compute FFT of an empty signal.")
    spectrum = fft(signal); freqs = fftfreq(n, d=1.0/sampling_rate)[:n//2]
    magnitude = (2.0/n)*np.abs(spectrum[:n//2]); magnitude[0] /= 2.0
    power = magnitude**2; total = float(np.sum(power))
    idx = int(np.argmax(magnitude)); centroid = float(np.sum(freqs*power)/total) if total else 0.0
    bandwidth = float(np.sqrt(np.sum((freqs-centroid)**2*power)/total)) if total else 0.0
    return FFTResult(freqs, magnitude, float(freqs[idx]), centroid, bandwidth, float(np.sum(power[freqs < low_cutoff])/total) if total else 0.0, float(np.sum(power[freqs >= high_cutoff])/total) if total else 0.0)
