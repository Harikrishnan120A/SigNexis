"""End-to-end pipeline test for SigNexis."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("END-TO-END PIPELINE TEST")
print("=" * 60)

# === TEST 1: 1kHz sine, 8kHz sampling, HF noise ===
print("\n[TEST 1] 1 kHz sine, fs=8000, HIGH_FREQUENCY_NOISE")
from dsp.signal_generator import generate_signal
t, clean, noisy = generate_signal(
    signal_type="sine", frequency=1000, amplitude=1.0,
    duration=1.0, sampling_rate=8000,
    noise_type="HIGH_FREQUENCY_NOISE", noise_amplitude=0.3,
    rng=np.random.default_rng(42)
)
print(f"  Signal samples: {len(noisy)}, t range: {t[0]:.4f}-{t[-1]:.4f} s")

# FFT
from dsp.fft_analysis import compute_fft
fft_r = compute_fft(noisy, 8000)
print(f"  Dominant frequency: {fft_r.dominant_freq:.1f} Hz  (expected ~1000 Hz)")
assert abs(fft_r.dominant_freq - 1000) < 50, f"FFT failed: dominant={fft_r.dominant_freq}"
print("  FFT PASSED")

# Features
from dsp.features import extract_features, FEATURE_NAMES, NUM_FEATURES
feats = extract_features(noisy, 8000)
assert len(feats) == NUM_FEATURES
assert all(np.isfinite(v) for v in feats.values())
print(f"  Features extracted: {NUM_FEATURES} features, all finite - PASSED")

# ML Prediction
from ml.predict import predict_signal
pred = predict_signal(noisy, 8000)
print(f"  ML Prediction: {pred.predicted_class} ({pred.confidence*100:.1f}% confidence)")
valid_classes = ["CLEAN", "GAUSSIAN_NOISE", "LOW_FREQUENCY_NOISE",
                 "HIGH_FREQUENCY_NOISE", "POWER_LINE_INTERFERENCE", "DISTORTED_SIGNAL"]
assert pred.predicted_class in valid_classes
print("  ML PASSED")

# Filter
from dsp.filters import apply_recommended_filter
filtered, finfo = apply_recommended_filter(noisy, 8000, pred.predicted_class, 1000)
assert len(filtered) == len(noisy)
ftype = finfo["filter_type"]
cutoff = finfo.get("cutoff_used")
print(f"  Filter applied: {ftype} at {cutoff} Hz")
print("  Filter PASSED")

# Quality
from dsp.quality import compute_quality, snr_improvement
qb = compute_quality(noisy, 8000, clean)
qa = compute_quality(filtered, 8000, clean)
print(f"  SNR before: {qb.snr_db:.2f} dB, after: {qa.snr_db:.2f} dB")
imp = snr_improvement(qb, qa)
print(f"  SNR improvement: {imp:+.2f} dB")
print("  Quality PASSED")

# === TEST 2: Aliasing check ===
print("\n[TEST 2] Aliasing detection: f=5000 Hz, fs=8000 Hz")
from dsp.sampling import analyse_sampling
result = analyse_sampling(5000, 8000)
print(f"  Nyquist: {result.nyquist_freq} Hz")
print(f"  Aliasing detected: {result.aliasing_detected}")
print(f"  Apparent freq: {result.apparent_freq:.1f} Hz")
assert result.aliasing_detected is True
assert result.apparent_freq is not None
print("  ALIASING TEST PASSED")

# === TEST 3: Properly sampled ===
print("\n[TEST 3] Proper sampling: f=1000 Hz, fs=8000 Hz")
result2 = analyse_sampling(1000, 8000)
assert result2.is_properly_sampled is True
assert result2.aliasing_detected is False
print("  Status: Properly sampled - PASSED")

# === TEST 4: Undersampled (f == fs) ===
print("\n[TEST 4] Undersampling: f=1000 Hz, fs=1000 Hz (equal)")
result3 = analyse_sampling(1000, 1000)
assert result3.is_properly_sampled is False
print("  Status: Undersampled detected - PASSED")

print("\n" + "=" * 60)
print("ALL END-TO-END TESTS PASSED!")
print("=" * 60)
