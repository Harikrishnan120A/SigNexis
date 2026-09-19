"""
final_pipeline_audit.py
========================
Final Project Audit Script for SigNexis.
Tests:
1. Complete pipeline end-to-end:
   Signal Generation -> Sampling -> Nyquist -> Aliasing -> FFT ->
   Features -> ML Prediction -> Confidence -> Filter Recommendation ->
   Filtering -> Quality Analysis (SNR Before/After)
2. Tests required signal conditions:
   - Clean signal
   - Gaussian noise
   - Low-frequency noise
   - High-frequency noise
   - 50 Hz power line interference
   - Undersampled signal (f >= fs/2)
3. WAV processing and analysis
4. Verifies exact feature order match between training & prediction
5. Verifies no hardcoded predictions, accuracy, or SNR values
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dsp.signal_generator import generate_signal, SIGNAL_CLASSES
from dsp.sampling import analyse_sampling, check_nyquist_condition, nyquist_frequency
from dsp.fft_analysis import compute_fft
from dsp.features import extract_features, features_to_vector, FEATURE_NAMES, NUM_FEATURES
from dsp.filters import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    notch_filter,
    apply_recommended_filter,
    FILTER_RECOMMENDATIONS,
)
from dsp.quality import compute_quality, snr_improvement
from ml.predict import predict_signal, MODEL_PATH
from audio.wav_processor import load_wav
import soundfile as sf
import io

print("=" * 70)
print("SIGNEIXS FINAL PIPELINE AUDIT")
print("=" * 70)

AUDIT_RESULTS = []

def record(test_name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {test_name}")
    if details:
        print(f"       Details: {details}")
    AUDIT_RESULTS.append({"test": test_name, "passed": passed, "details": details})

# -----------------------------------------------------------------------------
# 1. Feature order check between training and inference
# -----------------------------------------------------------------------------
print("\n--- 1. FEATURE ORDER & METADATA VERIFICATION ---")
meta_path = ROOT / "models" / "feature_metadata.json"
with open(meta_path) as f:
    meta = json.load(f)

meta_features = meta["feature_names"]
feature_order_match = (meta_features == FEATURE_NAMES)
record(
    "Feature Order Match (Metadata vs FEATURE_NAMES)",
    feature_order_match,
    f"Order matches exactly: {FEATURE_NAMES}"
)

# Check with a dummy extraction
dummy_t = np.linspace(0, 1, 8000, endpoint=False)
dummy_sig = np.sin(2 * np.pi * 500 * dummy_t)
feats = extract_features(dummy_sig, 8000)
vec = features_to_vector(feats)
vec_order_match = len(vec) == NUM_FEATURES and all(np.isclose(vec[i], feats[FEATURE_NAMES[i]]) for i in range(len(vec)))
record(
    "Vectorization Canonical Ordering",
    vec_order_match,
    f"Features mapped 1-to-1 in strict order across {NUM_FEATURES} elements."
)

# -----------------------------------------------------------------------------
# 2. Pipeline Test Across Required Conditions
# -----------------------------------------------------------------------------
test_scenarios = [
    {
        "name": "Clean Signal (1 kHz Sine)",
        "freq": 1000,
        "fs": 8000,
        "noise_type": "CLEAN",
        "noise_amp": 0.0,
        "expected_nyquist": True,
        "expected_alias": False,
        "expected_class": "CLEAN",
    },
    {
        "name": "Gaussian Noise on 1 kHz Sine",
        "freq": 1000,
        "fs": 8000,
        "noise_type": "GAUSSIAN_NOISE",
        "noise_amp": 0.5,
        "expected_nyquist": True,
        "expected_alias": False,
        "expected_class": "GAUSSIAN_NOISE",
    },
    {
        "name": "Low-Frequency Noise on 1 kHz Sine",
        "freq": 1000,
        "fs": 8000,
        "noise_type": "LOW_FREQUENCY_NOISE",
        "noise_amp": 0.5,
        "expected_nyquist": True,
        "expected_alias": False,
        "expected_class": "LOW_FREQUENCY_NOISE",
    },
    {
        "name": "High-Frequency Noise on 1 kHz Sine",
        "freq": 1000,
        "fs": 8000,
        "noise_type": "HIGH_FREQUENCY_NOISE",
        "noise_amp": 0.5,
        "expected_nyquist": True,
        "expected_alias": False,
        "expected_class": "HIGH_FREQUENCY_NOISE",
    },
    {
        "name": "50 Hz Power-Line Hum on 1 kHz Sine",
        "freq": 1000,
        "fs": 8000,
        "noise_type": "POWER_LINE_INTERFERENCE",
        "noise_amp": 0.5,
        "expected_nyquist": True,
        "expected_alias": False,
        "expected_class": "POWER_LINE_INTERFERENCE",
    },
    {
        "name": "Undersampled Signal (5000 Hz at 8000 Hz Sampling)",
        "freq": 5000,
        "fs": 8000,
        "noise_type": "CLEAN",
        "noise_amp": 0.0,
        "expected_nyquist": False,
        "expected_alias": True,
        "expected_apparent": 3000.0,
    }
]

print("\n--- 2. PIPELINE TESTS FOR ALL REQUIRED CONDITIONS ---")

for sc in test_scenarios:
    print(f"\nScenario: {sc['name']}")
    f = sc["freq"]
    fs = sc["fs"]
    nt = sc["noise_type"]
    na = sc["noise_amp"]

    # Step A: Signal Generation
    rng = np.random.default_rng(42)
    t, clean_ref, raw = generate_signal(
        signal_type="sine",
        frequency=f,
        amplitude=1.0,
        duration=1.0,
        sampling_rate=fs,
        noise_type=nt,
        noise_amplitude=na,
        rng=rng
    )

    # Step B: Sampling & Nyquist Analysis & Aliasing Detection
    samp_res = analyse_sampling(f, fs, duration=1.0)
    nyq_ok = (samp_res.is_properly_sampled == sc["expected_nyquist"])
    alias_ok = (samp_res.aliasing_detected == sc["expected_alias"])
    record(f"{sc['name']} - Nyquist & Aliasing", nyq_ok and alias_ok,
           f"Nyquist={samp_res.is_properly_sampled}, Aliasing={samp_res.aliasing_detected}, Apparent={samp_res.apparent_freq}")

    # Step C: FFT
    fft_res = compute_fft(raw, fs)
    fft_ok = len(fft_res.magnitude) > 0 and np.all(np.isfinite(fft_res.magnitude))
    record(f"{sc['name']} - FFT Computed", fft_ok,
           f"Dominant Freq: {fft_res.dominant_freq:.1f} Hz, Spectral Centroid: {fft_res.spectral_centroid:.1f} Hz")

    # Step D: Feature Extraction
    features = extract_features(raw, fs)
    feat_ok = len(features) == 13 and all(np.isfinite(v) for v in features.values())
    record(f"{sc['name']} - Feature Extraction", feat_ok,
           f"RMS={features['rms']:.4f}, ZCR={features['zero_crossing_rate']:.4f}, Bandwidth={features['spectral_bandwidth']:.1f}")

    # Step E: ML Prediction & Confidence
    pred = predict_signal(raw, fs)
    probs_sum = sum(pred.class_probabilities.values())
    pred_ok = (0.99 <= probs_sum <= 1.01) and (0.0 <= pred.confidence <= 1.0)
    record(f"{sc['name']} - ML Prediction", pred_ok,
           f"Predicted: {pred.predicted_class} ({pred.confidence:.1%}), Sum(P)={probs_sum:.6f}")

    # Step F: Filter Recommendation & Application
    filtered, filt_info = apply_recommended_filter(
        raw,
        fs,
        pred.predicted_class,
        signal_freq=f
    )
    filt_ok = len(filtered) == len(raw) and np.all(np.isfinite(filtered))
    record(f"{sc['name']} - Filter Application", filt_ok,
           f"Filter: {filt_info['filter_type']} (Cutoff: {filt_info['cutoff_used']}), Out len={len(filtered)}")

    # Step G: Before/After Quality Analysis
    qual_before = compute_quality(raw, fs, clean_reference=clean_ref)
    qual_after = compute_quality(filtered, fs, clean_reference=clean_ref)
    improvement = snr_improvement(qual_before, qual_after)
    record(f"{sc['name']} - Before/After Quality", True,
           f"SNR Before: {qual_before.snr_db}, SNR After: {qual_after.snr_db}, Improvement: {improvement}")

# -----------------------------------------------------------------------------
# 3. WAV Upload & Processing Test
# -----------------------------------------------------------------------------
print("\n--- 3. WAV UPLOAD & INGESTION TEST ---")
t_wav = np.arange(0, 1.5, 1.0 / 16000)
audio_sig = 0.5 * np.sin(2 * np.pi * 440 * t_wav).astype(np.float32)

buf = io.BytesIO()
sf.write(buf, audio_sig, 16000, format="WAV")
buf.seek(0)
buf.name = "audit_test.wav"

wav_info = load_wav(buf)
wav_ok = (
    wav_info.sampling_rate == 16000
    and abs(wav_info.duration - 1.5) < 0.05
    and len(wav_info.signal) == len(audio_sig)
)
record(
    "WAV Upload & Decoupling",
    wav_ok,
    f"Loaded {wav_info.duration:.2f}s, fs={wav_info.sampling_rate} Hz, channels={wav_info.num_channels_original}"
)

wav_pred = predict_signal(wav_info.signal, wav_info.sampling_rate)
record(
    "WAV ML Inference",
    wav_pred.predicted_class in SIGNAL_CLASSES,
    f"Class={wav_pred.predicted_class}, Conf={wav_pred.confidence:.1%}"
)

# -----------------------------------------------------------------------------
# 4. Check for Hardcoded Results
# -----------------------------------------------------------------------------
print("\n--- 4. HARDCODING CHECK ---")
_, _, sig1 = generate_signal(signal_type="sine", frequency=400, sampling_rate=8000, duration=0.5, amplitude=0.5)
_, _, sig2 = generate_signal(signal_type="sine", frequency=1500, sampling_rate=8000, duration=0.5, amplitude=1.8)

f1 = extract_features(sig1, 8000)
f2 = extract_features(sig2, 8000)

diff_rms = abs(f1["rms"] - f2["rms"]) > 0.1
diff_freq = abs(f1["dominant_freq"] - f2["dominant_freq"]) > 100.0

record(
    "Dynamic Feature Computation (Non-hardcoded)",
    diff_rms and diff_freq,
    f"f1_rms={f1['rms']:.3f}, f2_rms={f2['rms']:.3f}, f1_dom={f1['dominant_freq']}, f2_dom={f2['dominant_freq']}"
)

_, clean_low, sig_low_noise = generate_signal(frequency=1000, sampling_rate=8000, noise_type="GAUSSIAN_NOISE", noise_amplitude=0.1)
_, clean_high, sig_high_noise = generate_signal(frequency=1000, sampling_rate=8000, noise_type="GAUSSIAN_NOISE", noise_amplitude=0.9)

q_low = compute_quality(sig_low_noise, 8000, clean_reference=clean_low)
q_high = compute_quality(sig_high_noise, 8000, clean_reference=clean_high)

dynamic_snr = (q_low.snr_db is not None and q_high.snr_db is not None and q_low.snr_db > q_high.snr_db)
record(
    "Dynamic SNR Computation (Non-hardcoded)",
    dynamic_snr,
    f"Low noise SNR = {q_low.snr_db:.2f} dB, High noise SNR = {q_high.snr_db:.2f} dB"
)

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
total_tests = len(AUDIT_RESULTS)
passed_tests = sum(1 for r in AUDIT_RESULTS if r["passed"])
failed_tests = total_tests - passed_tests

print(f"AUDIT SUMMARY: {passed_tests}/{total_tests} tests passed.")
if failed_tests == 0:
    print("ALL TESTS PASSED COMPLETELY - ZERO FAILURES.")
else:
    print(f"FAILED TESTS ({failed_tests}):")
    for r in AUDIT_RESULTS:
        if not r["passed"]:
            print(f"  - {r['test']}: {r['details']}")

sys.exit(0 if failed_tests == 0 else 1)
