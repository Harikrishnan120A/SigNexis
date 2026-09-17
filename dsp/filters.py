import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos

def _validate(c, nyq, label):
    if c <= 0 or c >= nyq: raise ValueError(f"{label} must be between 0 and Nyquist.")
def low_pass_filter(signal, cutoff, sampling_rate, order=5):
    nyq=sampling_rate/2; _validate(cutoff, nyq, "LPF cutoff"); return sosfiltfilt(butter(order, cutoff/nyq, btype="low", output="sos"), signal)
def high_pass_filter(signal, cutoff, sampling_rate, order=5):
    nyq=sampling_rate/2; _validate(cutoff, nyq, "HPF cutoff"); return sosfiltfilt(butter(order, cutoff/nyq, btype="high", output="sos"), signal)
def band_pass_filter(signal, low_cutoff, high_cutoff, sampling_rate, order=5):
    nyq=sampling_rate/2; _validate(low_cutoff, nyq, "BPF low"); _validate(high_cutoff, nyq, "BPF high")
    if low_cutoff >= high_cutoff: raise ValueError("Low cutoff must be less than high cutoff.")
    return sosfiltfilt(butter(order, [low_cutoff/nyq, high_cutoff/nyq], btype="band", output="sos"), signal)
def notch_filter(signal, notch_freq, sampling_rate, quality_factor=30.0):
    nyq=sampling_rate/2; _validate(notch_freq, nyq, "Notch frequency"); b,a=iirnotch(notch_freq/nyq, quality_factor); return sosfiltfilt(tf2sos(b,a), signal)
FILTER_RECOMMENDATIONS={"CLEAN":{"filter_type":"none","reason":"Signal is classified as clean.","params":{}},"GAUSSIAN_NOISE":{"filter_type":"low_pass","reason":"Low-pass filtering attenuates broadband noise.","params":{"cutoff_fraction":.4}},"LOW_FREQUENCY_NOISE":{"filter_type":"high_pass","reason":"High-pass filtering removes baseline drift.","params":{"cutoff_hz":80.0}},"HIGH_FREQUENCY_NOISE":{"filter_type":"low_pass","reason":"Low-pass filtering attenuates high-frequency noise.","params":{"cutoff_fraction":.3}},"POWER_LINE_INTERFERENCE":{"filter_type":"notch","reason":"A 50 Hz notch removes mains hum.","params":{"notch_freq":50.0,"quality_factor":30.0}},"DISTORTED_SIGNAL":{"filter_type":"low_pass","reason":"Low-pass filtering smooths harmonic artefacts.","params":{"cutoff_fraction":.35}}}
def apply_recommended_filter(signal, sampling_rate, noise_class, signal_freq=None):
    rec=FILTER_RECOMMENDATIONS.get(noise_class, FILTER_RECOMMENDATIONS["GAUSSIAN_NOISE"]); info=dict(rec); nyq=sampling_rate/2; info["cutoff_used"]=None
    if rec["filter_type"]=="none": return signal.copy(),info
    if rec["filter_type"]=="notch": info["cutoff_used"]=50.0; return notch_filter(signal,50.0,sampling_rate,30.0),info
    if rec["filter_type"]=="high_pass": c=min(80.0,nyq*.9); info["cutoff_used"]=c; return high_pass_filter(signal,c,sampling_rate),info
    c=min(nyq*rec["params"]["cutoff_fraction"],nyq*.9); info["cutoff_used"]=c; return low_pass_filter(signal,c,sampling_rate),info
