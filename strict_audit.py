"""
strict_audit.py
===============
Strict ECE + ML technical audit of SigNexis.
Tests mathematical correctness of every DSP and ML component.
"""
import sys, json, warnings
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

PASS = []
FAIL = []
WARN = []

def ok(label):
    print(f"  [PASS] {label}")
    PASS.append(label)

def fail(label, detail=""):
    print(f"  [FAIL] {label}")
    if detail:
        print(f"         --> {detail}")
    FAIL.append(f"{label}: {detail}")

def warn(label, detail=""):
    print(f"  [WARN] {label}")
    if detail:
        print(f"         --> {detail}")
    WARN.append(f"{label}: {detail}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 1: NYQUIST / SAMPLING THEOREM")
print("="*70)

from dsp.sampling import analyse_sampling, nyquist_frequency, check_nyquist_condition

# 1a. Nyquist frequency formula: f_N = f_s / 2
fs = 8000
fn = nyquist_frequency(fs)
if fn == 4000.0:
    ok("f_N = f_s / 2 = 4000 Hz for fs=8000")
else:
    fail("Nyquist formula", f"Expected 4000, got {fn}")

# 1b. Properly sampled: f < f_N
if check_nyquist_condition(3999, 8000):
    ok("f=3999 Hz < f_N=4000 Hz  => properly sampled")
else:
    fail("Nyquist condition (f < f_N)", "3999 < 4000 should be True")

# 1c. Exactly at Nyquist: f == f_N should be UNDER-sampled
# Nyquist theorem: need fs > 2*f, i.e. f < fs/2. Equal is NOT sufficient.
if not check_nyquist_condition(4000, 8000):
    ok("f=f_N (4000 Hz) => correctly flagged as NOT properly sampled")
else:
    fail("Nyquist condition (f == f_N)", "Equal to Nyquist must NOT be properly sampled")

# 1d. Aliasing calculation: 5000 Hz @ 8000 Hz fs
#    f_alias = |f - round(f/fs)*fs|  => 5000 mod 8000 = 5000 > 4000, reflect: 8000-5000=3000
r = analyse_sampling(5000, 8000)
if r.aliasing_detected:
    ok("f=5000 Hz, fs=8000 Hz => aliasing detected")
else:
    fail("Aliasing detection", "5000 Hz > Nyquist 4000 Hz must cause aliasing")

expected_alias = 3000.0
if r.apparent_freq is not None and abs(r.apparent_freq - expected_alias) < 1.0:
    ok(f"Alias frequency = {r.apparent_freq:.1f} Hz  (expected {expected_alias:.1f} Hz)")
else:
    fail("Alias frequency calculation", f"Expected {expected_alias}, got {r.apparent_freq}")

# 1e. Aliasing: 5000 Hz @ 6000 Hz fs => alias = 5000 mod 6000 = 5000; 5000>3000 => 6000-5000=1000
r2 = analyse_sampling(5000, 6000)
expected_alias_2 = 1000.0
if r2.apparent_freq is not None and abs(r2.apparent_freq - expected_alias_2) < 1.0:
    ok(f"Alias frequency: f=5000, fs=6000 => {r2.apparent_freq:.1f} Hz (expected 1000 Hz)")
else:
    fail("Alias frequency (5000/6000)", f"Expected 1000, got {r2.apparent_freq}")

# 1f. Aliasing: 3*fs/2 alias => 12000 Hz @ 8000 Hz => alias = 12000 mod 8000 = 4000 == f_N
#    reflect: 8000 - 4000 = 4000. Apparent = 4000 (DC-adjacent case)
r3 = analyse_sampling(12000, 8000)
expected_alias_3 = 4000.0
if r3.apparent_freq is not None and abs(r3.apparent_freq - expected_alias_3) < 1.0:
    ok(f"Alias: f=12000, fs=8000 => {r3.apparent_freq:.1f} Hz (expected {expected_alias_3} Hz)")
else:
    fail("Alias frequency (12000/8000)", f"Expected {expected_alias_3}, got {r3.apparent_freq}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 2: FFT CORRECTNESS")
print("="*70)

from dsp.fft_analysis import compute_fft

# 2a. Pure sine - dominant freq matches input
fs = 8000; f0 = 1000; N = 8000
t = np.arange(N) / fs
x = np.sin(2 * np.pi * f0 * t)
r = compute_fft(x, fs)
if abs(r.dominant_freq - f0) < (fs / N):   # within one frequency bin
    ok(f"Pure 1 kHz sine: dominant_freq = {r.dominant_freq:.1f} Hz (expected {f0})")
else:
    fail("FFT dominant frequency", f"Expected {f0}, got {r.dominant_freq:.1f}")

# 2b. Frequency axis max = Nyquist
if abs(r.freqs[-1] - (fs/2 - fs/N)) < 1.0:
    ok(f"Frequency axis upper bound ≈ Nyquist ({r.freqs[-1]:.1f} Hz)")
else:
    fail("Frequency axis upper bound", f"Expected ~{fs/2}, got {r.freqs[-1]:.1f}")

# 2c. DC bin: pure sine centered on zero should have near-zero mean/DC
#     For a sine wave of exactly integer cycles, DC component should be ~0
dc_magnitude = r.magnitude[0]
if dc_magnitude < 0.01:
    ok(f"DC magnitude for pure sine ≈ {dc_magnitude:.6f} (near zero, correct)")
else:
    warn("DC magnitude", f"DC={dc_magnitude:.4f} higher than expected for a pure sine")

# 2d. Magnitude normalisation: for A*sin(2*pi*f*t), single-sided amplitude ≈ A
# The magnitude at the dominant bin should be ≈ 1.0 (amplitude of the sine)
peak_mag = r.magnitude[np.argmax(r.magnitude)]
if abs(peak_mag - 1.0) < 0.05:
    ok(f"FFT peak magnitude = {peak_mag:.4f} (expected 1.0 for A=1 sine)")
else:
    fail("FFT amplitude normalisation",
         f"Expected peak magnitude ~1.0, got {peak_mag:.4f}. "
         f"Check the 2/N normalisation factor.")

# 2e. Spectral centroid for a pure tone must equal dominant freq
if abs(r.spectral_centroid - f0) < 10:
    ok(f"Spectral centroid ≈ {r.spectral_centroid:.1f} Hz matches dominant {f0} Hz")
else:
    warn("Spectral centroid vs dominant freq",
         f"Centroid={r.spectral_centroid:.1f} Hz, dominant={f0} Hz. "
         f"Power of noise floor causes drift")

# 2f. Band energy fractions must sum ≤ 1.0
total_energy = r.low_freq_energy + r.high_freq_energy
if total_energy <= 1.0 + 1e-9:
    ok(f"low_freq_energy + high_freq_energy = {total_energy:.4f} ≤ 1.0")
else:
    fail("Band energy fractions", f"Sum = {total_energy:.4f} > 1.0")

# 2g. 500 Hz sine in 8000 Hz fs
f0b = 500
xb = np.sin(2 * np.pi * f0b * t)
rb = compute_fft(xb, fs)
if abs(rb.dominant_freq - f0b) < (fs / N):
    ok(f"500 Hz sine: dominant_freq = {rb.dominant_freq:.1f} Hz")
else:
    fail("FFT 500 Hz", f"Expected 500, got {rb.dominant_freq:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 3: FEATURE EXTRACTION CORRECTNESS")
print("="*70)

from dsp.features import extract_features, features_to_vector, FEATURE_NAMES, NUM_FEATURES

# 3a. RMS of sine A*sin(2*pi*f*t) = A/sqrt(2) = 0.7071
t = np.arange(8000) / 8000
x = np.sin(2 * np.pi * 1000 * t)
feats = extract_features(x, 8000)
expected_rms = 1.0 / np.sqrt(2)
if abs(feats["rms"] - expected_rms) < 0.01:
    ok(f"RMS of unit sine = {feats['rms']:.5f}  (expected {expected_rms:.5f})")
else:
    fail("RMS formula", f"Expected {expected_rms:.5f}, got {feats['rms']:.5f}")

# 3b. Signal power = rms^2 = 0.5 for unit sine
expected_power = 0.5
if abs(feats["signal_power"] - expected_power) < 0.01:
    ok(f"Signal power = {feats['signal_power']:.5f}  (expected {expected_power:.5f})")
else:
    fail("Signal power formula", f"Expected {expected_power:.5f}, got {feats['signal_power']:.5f}")

# 3c. Mean of a pure sine ≈ 0 (for integer cycles)
if abs(feats["mean"]) < 0.001:
    ok(f"Mean of pure sine ≈ 0  ({feats['mean']:.6f})")
else:
    warn("Mean of pure sine", f"Non-zero mean: {feats['mean']:.6f}")

# 3d. Peak-to-peak = max - min. For unit sine: max=1, min=-1, p2p=2
if abs(feats["peak_to_peak"] - 2.0) < 0.01:
    ok(f"Peak-to-peak = {feats['peak_to_peak']:.5f}  (expected 2.0)")
else:
    fail("Peak-to-peak formula", f"Expected 2.0, got {feats['peak_to_peak']:.5f}")

# 3e. ZCR: 1000 Hz sine @ 8000 Hz → 2 zero-crossings per cycle → 2000 crossings / 7999 ≈ 0.25
expected_zcr = 2 * 1000.0 / 8000  # 2 crossings per cycle * f / fs = 0.25
if abs(feats["zero_crossing_rate"] - expected_zcr) < 0.02:
    ok(f"Zero crossing rate = {feats['zero_crossing_rate']:.4f}  (expected ~{expected_zcr:.4f})")
else:
    warn("Zero crossing rate", f"Expected ~{expected_zcr:.4f}, got {feats['zero_crossing_rate']:.4f}")

# 3f. Features-to-vector: order matches FEATURE_NAMES
vec = features_to_vector(feats)
if len(vec) == NUM_FEATURES:
    ok(f"Feature vector length = {NUM_FEATURES} (matches NUM_FEATURES)")
else:
    fail("Feature vector length", f"Expected {NUM_FEATURES}, got {len(vec)}")

for i, name in enumerate(FEATURE_NAMES):
    if abs(vec[i] - feats[name]) > 1e-10:
        fail(f"Feature vector order mismatch at index {i}", f"Expected {name}={feats[name]}, got {vec[i]}")
        break
else:
    ok("Feature vector canonical order matches FEATURE_NAMES")

# 3g. No NaN in features for noisy signal
from dsp.signal_generator import generate_signal
_, _, noisy = generate_signal(noise_type="GAUSSIAN_NOISE", noise_amplitude=0.5,
                               sampling_rate=8000, frequency=1000, rng=np.random.default_rng(0))
feats_noisy = extract_features(noisy, 8000)
if all(np.isfinite(v) for v in feats_noisy.values()):
    ok("No NaN/Inf in features for GAUSSIAN_NOISE signal")
else:
    fail("NaN in features", f"Non-finite features: {[k for k,v in feats_noisy.items() if not np.isfinite(v)]}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 4: FILTER CORRECTNESS")
print("="*70)

from dsp.filters import low_pass_filter, high_pass_filter, notch_filter, band_pass_filter

fs = 8000
t = np.arange(8000) / fs

# 4a. LPF: 1 kHz sine passes, 3.5 kHz is attenuated
sig_low = np.sin(2 * np.pi * 1000 * t)
sig_high = np.sin(2 * np.pi * 3500 * t)
mixed = sig_low + sig_high
filtered = low_pass_filter(mixed, cutoff=2000.0, sampling_rate=fs)

rms_original_high = float(np.sqrt(np.mean(sig_high**2)))
# Check that 3.5 kHz is attenuated in the filtered output
from dsp.fft_analysis import compute_fft as cfft
fft_f = cfft(filtered, fs)
fft_m = cfft(mixed, fs)
idx_high = np.argmin(np.abs(fft_f.freqs - 3500))
idx_1k   = np.argmin(np.abs(fft_f.freqs - 1000))
ratio = fft_f.magnitude[idx_high] / (fft_f.magnitude[idx_1k] + 1e-10)
if ratio < 0.1:
    ok(f"LPF @ 2 kHz: 3.5 kHz component attenuated (ratio {ratio:.4f} < 0.1)")
else:
    fail("LPF attenuation", f"3.5 kHz not sufficiently attenuated: ratio={ratio:.4f}")

# 4b. HPF: 100 Hz component removed, 1 kHz passes
sig_drift = 0.5 * np.sin(2 * np.pi * 20 * t)
sig_sig   = np.sin(2 * np.pi * 1000 * t)
mixed_h = sig_drift + sig_sig
filtered_h = high_pass_filter(mixed_h, cutoff=200.0, sampling_rate=fs)
fft_h_in = cfft(mixed_h, fs)
fft_h_out = cfft(filtered_h, fs)
idx_20  = np.argmin(np.abs(fft_h_out.freqs - 20))
idx_1k2 = np.argmin(np.abs(fft_h_out.freqs - 1000))
ratio_h = fft_h_out.magnitude[idx_20] / (fft_h_out.magnitude[idx_1k2] + 1e-10)
if ratio_h < 0.05:
    ok(f"HPF @ 200 Hz: 20 Hz drift attenuated (ratio {ratio_h:.4f} < 0.05)")
else:
    fail("HPF attenuation", f"20 Hz component not removed: ratio={ratio_h:.4f}")

# 4c. Notch @ 50 Hz: removes 50 Hz hum
sig_50 = np.sin(2 * np.pi * 50 * t)
sig_sig2 = np.sin(2 * np.pi * 1000 * t)
mixed_n = sig_sig2 + 0.5 * sig_50
filtered_n = notch_filter(mixed_n, notch_freq=50.0, sampling_rate=fs, quality_factor=30)
fft_n_out = cfft(filtered_n, fs)
idx_50  = np.argmin(np.abs(fft_n_out.freqs - 50))
idx_1k3 = np.argmin(np.abs(fft_n_out.freqs - 1000))
ratio_n = fft_n_out.magnitude[idx_50] / (fft_n_out.magnitude[idx_1k3] + 1e-10)
if ratio_n < 0.2:
    ok(f"Notch @ 50 Hz: 50 Hz attenuated (ratio {ratio_n:.4f} < 0.2)")
else:
    fail("Notch filter attenuation", f"50 Hz not removed: ratio={ratio_n:.4f}")

# 4d. Filter output length must equal input length
x_test = np.random.randn(1000)
for fname, func, kw in [
    ("LPF", low_pass_filter, dict(cutoff=500.0, sampling_rate=8000)),
    ("HPF", high_pass_filter, dict(cutoff=200.0, sampling_rate=8000)),
    ("Notch", notch_filter, dict(notch_freq=50.0, sampling_rate=8000)),
]:
    out = func(x_test, **kw)
    if len(out) == len(x_test):
        ok(f"{fname}: output length matches input ({len(x_test)})")
    else:
        fail(f"{fname} length mismatch", f"Input {len(x_test)}, output {len(out)}")

# 4e. Invalid cutoff raises
import pytest as pt_mod
try:
    low_pass_filter(np.ones(100), cutoff=4001, sampling_rate=8000)
    fail("LPF cutoff > Nyquist: should raise ValueError")
except ValueError:
    ok("LPF cutoff > Nyquist raises ValueError")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 5: SNR COMPUTATION")
print("="*70)

from dsp.quality import compute_quality, snr_improvement

# 5a. SNR with known noise
signal_clean = np.sin(2 * np.pi * 1000 * t)      # power = 0.5
noise = np.random.default_rng(0).normal(0, 0.1, len(t))  # power = sigma^2 = 0.01
noisy_signal = signal_clean + noise
qm = compute_quality(noisy_signal, 8000, signal_clean)
expected_snr = 10 * np.log10(0.5 / np.mean(noise**2))
if abs(qm.snr_db - expected_snr) < 1.0:
    ok(f"SNR = {qm.snr_db:.2f} dB  (expected ≈ {expected_snr:.2f} dB)")
else:
    fail("SNR formula", f"Expected {expected_snr:.2f} dB, got {qm.snr_db:.2f} dB")

# 5b. SNR = 10*log10(P_signal / P_noise) — verify formula
P_s = float(np.mean(signal_clean**2))
noise_comp = noisy_signal - signal_clean
P_n = float(np.mean(noise_comp**2))
expected = 10 * np.log10(P_s / P_n)
if abs(qm.snr_db - expected) < 0.001:
    ok("SNR = 10*log10(P_signal/P_noise) formula verified")
else:
    fail("SNR formula consistency", f"Manual {expected:.4f} != code {qm.snr_db:.4f}")

# 5c. Clean signal => SNR = inf or very large
qm_clean = compute_quality(signal_clean, 8000, signal_clean)
if qm_clean.snr_db == float("inf") or qm_clean.snr_db > 100:
    ok(f"Clean signal SNR = {qm_clean.snr_db} (correct: infinity or very large)")
else:
    fail("SNR for clean signal", f"Expected inf, got {qm_clean.snr_db}")

# 5d. No reference => SNR = None (not a fake number)
qm_no_ref = compute_quality(noisy_signal, 8000)
if qm_no_ref.snr_db is None:
    ok("No clean reference => SNR = None (correct, no fake estimate)")
else:
    fail("SNR without reference", f"Should be None, got {qm_no_ref.snr_db}")

# 5e. SNR improvement = after_snr - before_snr
# Apply LPF to reduce noise, check SNR actually improves
from dsp.filters import low_pass_filter
filtered_sig = low_pass_filter(noisy_signal, cutoff=2000, sampling_rate=8000)
qm_after = compute_quality(filtered_sig, 8000, signal_clean)
imp = snr_improvement(qm, qm_after)
if imp is not None and imp > 0:
    ok(f"SNR improvement after LPF: {imp:+.2f} dB  (positive = improvement)")
else:
    warn("SNR improvement", f"LPF on 10 dB SNR signal: improvement={imp}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 6: ML PIPELINE - DATA LEAKAGE CHECK")
print("="*70)

from ml.dataset import generate_dataset, DATASET_PATH
from ml.train import FEATURE_NAMES as train_feat_names

# 6a. Feature name alignment: training features == inference features
meta_path = ROOT / "models" / "feature_metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    saved_names = meta["feature_names"]
    if saved_names == FEATURE_NAMES:
        ok("Feature metadata matches FEATURE_NAMES (training = inference)")
    else:
        fail("Feature name mismatch", f"Metadata: {saved_names}\nCode: {FEATURE_NAMES}")
else:
    fail("feature_metadata.json missing", "Run python -m ml.dataset first")

# 6b. Check for train/test contamination: generate fresh small dataset, split
df_check = generate_dataset(samples_per_class=50, seed=99, verbose=False)
from sklearn.model_selection import train_test_split
X_c = df_check[FEATURE_NAMES].to_numpy(dtype=np.float64)
y_c = df_check["label"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)
# Check no overlap
# With fixed seed and no shuffling trick, row sets should not overlap
tr_set = set(map(tuple, X_tr))
te_set = set(map(tuple, X_te))
overlap = tr_set & te_set
if len(overlap) == 0:
    ok("No train/test data overlap (no leakage)")
else:
    fail("Train/test leakage detected", f"{len(overlap)} overlapping rows")

# 6c. Check that no label is in features (obvious label-leak)
if "label" not in FEATURE_NAMES:
    ok("Label column not included in feature vector (no label leakage)")
else:
    fail("Label leakage", "Label column is inside FEATURE_NAMES!")

# 6d. Feature scaling: RF does not need scaling, but check for data-dependent normalisation
# Scan features.py for any global statistics being saved/used
with open(ROOT / "dsp" / "features.py") as f:
    feat_src = f.read()
if "fit" not in feat_src and "StandardScaler" not in feat_src and "MinMaxScaler" not in feat_src:
    ok("No scaler fitted on training data that would cause test leakage")
else:
    warn("Scaler found in features.py", "Verify scaler is fit only on training data")

# 6e. Training uses stratified split
with open(ROOT / "ml" / "train.py") as f:
    train_src = f.read()
if "stratify=y" in train_src:
    ok("Stratified split used (stratify=y)")
else:
    fail("Stratified split missing", "Class imbalance not addressed by stratified split")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 7: ML MODEL - LOADED MODEL vs TRAINED MODEL CONSISTENCY")
print("="*70)

from ml.predict import predict_signal, load_model, MODEL_PATH

try:
    model = load_model(MODEL_PATH)
    # 7a. Model classes match SIGNAL_CLASSES
    from dsp.signal_generator import SIGNAL_CLASSES
    model_classes = list(model.classes_)
    if set(model_classes) == set(SIGNAL_CLASSES):
        ok(f"Model classes match SIGNAL_CLASSES: {model_classes}")
    else:
        fail("Model class mismatch",
             f"Model: {sorted(model_classes)}\nExpected: {sorted(SIGNAL_CLASSES)}")

    # 7b. Model input dimensionality matches NUM_FEATURES
    n_feat_model = model.n_features_in_
    if n_feat_model == NUM_FEATURES:
        ok(f"Model input features = {n_feat_model} (matches NUM_FEATURES={NUM_FEATURES})")
    else:
        fail("Feature dimensionality mismatch",
             f"Model expects {n_feat_model}, code provides {NUM_FEATURES}")

    # 7c. Model estimators count
    if hasattr(model, "n_estimators") and model.n_estimators == 200:
        ok(f"Model has {model.n_estimators} trees (as configured)")
    else:
        warn("Model estimators", f"Expected 200, got {getattr(model, 'n_estimators', '?')}")

    # 7d. Prediction output consistency
    t_test = np.arange(0, 0.5, 1.0 / 8000)
    x_test = np.sin(2 * np.pi * 1000 * t_test)
    pred = predict_signal(x_test, 8000)
    proba_sum = sum(pred.class_probabilities.values())
    if abs(proba_sum - 1.0) < 1e-6:
        ok(f"Class probabilities sum to 1.0 ({proba_sum:.8f})")
    else:
        fail("Probability sum", f"Expected 1.0, got {proba_sum:.8f}")

    # 7e. Confidence = probability of predicted class
    conf = pred.confidence
    class_prob = pred.class_probabilities[pred.predicted_class]
    if abs(conf - class_prob) < 1e-9:
        ok(f"Confidence = class_probability[predicted_class] ({conf:.4f})")
    else:
        fail("Confidence mismatch", f"conf={conf}, class_prob[predicted]={class_prob}")

except FileNotFoundError as e:
    fail("Model load", str(e))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 8: ML ACCURACY - CONFUSION MATRIX vs STORED RESULTS")
print("="*70)

eval_path = ROOT / "models" / "evaluation_results.json"
if eval_path.exists():
    with open(eval_path) as f:
        stored = json.load(f)
    acc = stored["accuracy"]
    cm = stored["confusion_matrix"]
    classes = stored["classes"]
    n_test = stored["n_test"]

    ok(f"Stored accuracy = {acc*100:.2f}%  on {n_test} test samples")

    # Recompute accuracy from confusion matrix
    cm_arr = np.array(cm)
    cm_acc = np.trace(cm_arr) / np.sum(cm_arr)
    if abs(cm_acc - acc) < 0.001:
        ok(f"Confusion matrix trace / total = {cm_acc:.4f} matches stored accuracy {acc:.4f}")
    else:
        fail("Accuracy consistency", f"From CM: {cm_acc:.4f}, stored: {acc:.4f}")

    # Per-class diagonal (correct classifications)
    print("\n  Confusion Matrix (rows=true, cols=pred):")
    header = "  " + "".join(f"{c[:6]:>8s}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "  " + f"{cls[:14]:14s}" + "".join(f"{cm_arr[i,j]:8d}" for j in range(len(classes)))
        print(row)

    # Check if any class has recall < 0.5 (serious problem)
    per_class_recall = cm_arr.diagonal() / cm_arr.sum(axis=1)
    poor_classes = [(cls, r) for cls, r in zip(classes, per_class_recall) if r < 0.5]
    if poor_classes:
        for cls, r in poor_classes:
            warn(f"Low recall for {cls}", f"Recall = {r:.2f} < 0.5")
    else:
        ok("All classes have recall >= 0.5")
else:
    warn("evaluation_results.json not found", "Run ml.train to generate it")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 9: SIGNAL GENERATOR CORRECTNESS")
print("="*70)

from dsp.signal_generator import generate_signal, generate_reference_signal

# 9a. Clean signal: noisy == clean
t, clean, noisy = generate_signal(noise_type="CLEAN", frequency=1000, sampling_rate=8000,
                                   rng=np.random.default_rng(0))
if np.allclose(clean, noisy):
    ok("CLEAN noise type: noisy signal = clean signal")
else:
    fail("CLEAN signal not identical to clean")

# 9b. Sampling rate check: len(t) == fs * duration for duration=1
t2, c2, n2 = generate_signal(frequency=500, sampling_rate=8000, duration=1.0,
                              noise_type="CLEAN", rng=np.random.default_rng(0))
if len(t2) == 8000:
    ok(f"Signal length = fs * duration = {len(t2)} samples (correct)")
else:
    fail("Signal length", f"Expected 8000, got {len(t2)}")

# 9c. DISTORTED_SIGNAL: clipping adds energy, but signal should differ from clean
t3, c3, n3 = generate_signal(noise_type="DISTORTED_SIGNAL", frequency=500,
                              sampling_rate=8000, amplitude=1.0,
                              rng=np.random.default_rng(0))
if not np.allclose(c3, n3):
    ok("DISTORTED_SIGNAL differs from clean")
else:
    fail("DISTORTED_SIGNAL identical to clean")

# 9d. HIGH_FREQUENCY_NOISE: validate it's actually HF (spectral energy > 1kHz)
t4, c4, n4 = generate_signal(noise_type="HIGH_FREQUENCY_NOISE", frequency=200,
                              sampling_rate=8000, noise_amplitude=1.0,
                              rng=np.random.default_rng(0))
fft_noise = cfft(n4 - c4, 8000)   # isolate noise
hf_frac = fft_noise.high_freq_energy
if hf_frac > 0.1:
    ok(f"HIGH_FREQUENCY_NOISE has HF energy fraction = {hf_frac:.3f} (> 0.1)")
else:
    warn("HIGH_FREQUENCY_NOISE HF fraction", f"Only {hf_frac:.3f} of energy above 2 kHz")

# 9e. Reference signal: should be close to signal_discrete at sample times
t_ref, sig_ref = generate_reference_signal(1000, 1.0, 0.01)
if len(t_ref) == int(192000 * 0.01):
    ok(f"Reference signal length correct ({len(t_ref)} samples at 192 kHz)")
else:
    fail("Reference signal length", f"Expected {int(192000*0.01)}, got {len(t_ref)}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 10: PRACTICAL NOISE DISTINGUISHABILITY CHECK")
print("="*70)
# This is a practical test: are the 6 classes actually distinguishable by features?
# Run a quick 3-fold CV on fresh data and check it beats chance (>16.7%)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RF

df_cv = generate_dataset(samples_per_class=100, seed=77, verbose=False)
X_cv = df_cv[FEATURE_NAMES].to_numpy(dtype=np.float64)
y_cv = df_cv["label"].to_numpy()
clf_cv = RF(n_estimators=50, random_state=0, n_jobs=-1)
scores = cross_val_score(clf_cv, X_cv, y_cv, cv=3, scoring="accuracy")
mean_acc = scores.mean()
chance_level = 1.0 / 6
if mean_acc > chance_level + 0.3:   # must be at least 30% above chance
    ok(f"3-fold CV accuracy = {mean_acc:.3f} ({mean_acc*100:.1f}%) >> chance {chance_level:.3f}")
else:
    fail("Classes not well-separated",
         f"CV accuracy {mean_acc:.3f} barely above chance {chance_level:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)
print(f"  PASS : {len(PASS)}")
print(f"  WARN : {len(WARN)}")
print(f"  FAIL : {len(FAIL)}")
print()
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"  - {f}")
if WARN:
    print("WARNINGS:")
    for w in WARN:
        print(f"  - {w}")
if not FAIL:
    print("All critical checks PASSED.")
