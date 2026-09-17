from dataclasses import dataclass
import numpy as np
from dsp.fft_analysis import compute_fft
@dataclass
class QualityMetrics:
    rms: float; peak_amplitude: float; dominant_freq: float; signal_power: float; noise_power: float | None; snr_db: float | None; snr_note: str
def compute_quality(signal, sampling_rate, clean_reference=None):
    if len(signal)==0: raise ValueError("Signal must not be empty.")
    power=float(np.mean(signal**2)); noise=None; snr=None
    if clean_reference is not None and len(clean_reference)==len(signal):
        noise=float(np.mean((signal-clean_reference)**2)); ref=float(np.mean(clean_reference**2)); snr=float(10*np.log10(ref/noise)) if noise>0 and ref>0 else (float("inf") if noise==0 else None)
    return QualityMetrics(float(np.sqrt(power)),float(np.max(np.abs(signal))),compute_fft(signal,sampling_rate).dominant_freq,power,noise,snr,"SNR uses the clean reference." if clean_reference is not None else "No clean reference available.")
def snr_improvement(before, after): return after.snr_db-before.snr_db if before.snr_db is not None and after.snr_db is not None else None
def estimate_snr_rms_ratio(noisy, filtered):
    a=np.sqrt(np.mean(noisy**2)); b=np.sqrt(np.mean(filtered**2)); return float(20*np.log10(a/b)) if a>0 and b>0 else 0.0
