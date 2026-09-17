from dataclasses import dataclass
import numpy as np
from dsp.signal_generator import generate_reference_signal
@dataclass
class SamplingAnalysis:
    sampling_rate: float; nyquist_freq: float; signal_freq: float; is_properly_sampled: bool; aliasing_detected: bool; apparent_freq: float | None; t_ref: np.ndarray; signal_ref: np.ndarray; t_discrete: np.ndarray; signal_discrete: np.ndarray; status_message: str = ""
    def __post_init__(self):
        if not self.status_message: self.status_message = "Properly sampled" if self.is_properly_sampled else "UNDERSAMPLED / Aliasing detected"
def analyse_sampling(signal_freq, sampling_rate, amplitude=1.0, duration=.05):
    if sampling_rate <= 0 or signal_freq <= 0: raise ValueError("Frequencies and sampling rate must be positive.")
    nyquist = sampling_rate/2; alias = signal_freq >= nyquist; folded = signal_freq % sampling_rate
    apparent = (folded if folded <= nyquist else sampling_rate-folded) if alias else None
    tref, ref = generate_reference_signal(signal_freq, amplitude, duration)
    td = np.arange(0, duration, 1/sampling_rate)
    return SamplingAnalysis(sampling_rate, nyquist, signal_freq, not alias, alias, apparent, tref, ref, td, amplitude*np.sin(2*np.pi*signal_freq*td))
def nyquist_frequency(sampling_rate): return sampling_rate/2
def check_nyquist_condition(signal_freq, sampling_rate): return signal_freq < nyquist_frequency(sampling_rate)
