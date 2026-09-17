import json
from pathlib import Path
import numpy as np
from dsp.fft_analysis import compute_fft
FEATURE_NAMES = ["mean", "std", "rms", "max_amplitude", "min_amplitude", "peak_to_peak", "zero_crossing_rate", "dominant_freq", "spectral_centroid", "spectral_bandwidth", "low_freq_energy", "high_freq_energy", "signal_power"]
NUM_FEATURES = len(FEATURE_NAMES)
def extract_features(signal, sampling_rate, clean_reference=None):
    if len(signal) == 0: raise ValueError("Signal must not be empty.")
    fft = compute_fft(signal, sampling_rate)
    values = [np.mean(signal), np.std(signal), np.sqrt(np.mean(signal**2)), np.max(signal), np.min(signal), np.ptp(signal), np.sum(np.diff(np.sign(signal)) != 0)/max(1, len(signal)-1), fft.dominant_freq, fft.spectral_centroid, fft.spectral_bandwidth, fft.low_freq_energy, fft.high_freq_energy, np.mean(signal**2)]
    return {name: float(value) if np.isfinite(value) else 0.0 for name, value in zip(FEATURE_NAMES, values)}
def features_to_vector(features): return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
def save_feature_metadata(path="models/feature_metadata.json"):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({"feature_names": FEATURE_NAMES, "num_features": NUM_FEATURES}, indent=2))
def load_feature_metadata(path="models/feature_metadata.json"): return json.loads(Path(path).read_text())
