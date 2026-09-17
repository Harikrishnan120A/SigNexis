# SigNexis
## ML-Based Intelligent Digital Signal Analysis, Noise Classification and Adaptive Filtering System

> **ECE Undergraduate Mini-Project** | Python · NumPy · SciPy · Scikit-learn · Streamlit · Plotly

---

## Problem Statement

Real-world signals are always contaminated by noise. Identifying the type of noise and selecting the correct filter is traditionally a manual, expert-driven process. SigNexis automates this by combining classical Digital Signal Processing with a trained Machine Learning model to:

1. Detect the noise / distortion condition from signal features
2. Recommend and apply the most appropriate digital filter
3. Quantify the improvement in signal quality

---

## Project Objectives

- Demonstrate Nyquist Sampling Theorem and aliasing interactively
- Implement FFT-based spectral analysis with correct mathematics
- Extract a unified DSP feature set used identically at training and inference
- Train a Random Forest classifier on synthetic signal data
- Build an engineering-grade Streamlit dashboard showing the complete DSP→ML pipeline

---

## System Architecture

```
INPUT SIGNAL
    │
    ▼
SIGNAL GENERATOR  ──────────────────────────────────────┐
    │  (sine / multi-sine, configurable f, A, fs, noise) │
    ▼                                                     │
SAMPLING MODULE                                          │
    │  Nyquist check, aliasing detection                 │
    ▼                                                     │
FFT ANALYSIS                                             │
    │  magnitude spectrum, spectral features             │
    ▼                                                     │
FEATURE EXTRACTION  (same pipeline: train + infer)       │
    │  13 time/freq-domain features                      │
    ▼                                                     │
RANDOM FOREST CLASSIFIER                                 │
    │  6 classes: CLEAN, GAUSSIAN, LF, HF, PLI, DIST    │
    ▼                                                     │
NOISE CLASSIFICATION + CONFIDENCE                        │
    │                                                     │
    ▼                                                     │
FILTER RECOMMENDATION ENGINE  ◄──────────────────────────┘
    │  LPF / HPF / Notch / BPF
    ▼
DIGITAL FILTER  (scipy.signal Butterworth / IIR notch)
    │
    ▼
SIGNAL QUALITY ANALYSIS
    │  RMS, peak, SNR (with reference), SNR improvement
    ▼
BEFORE / AFTER COMPARISON  (waveform + FFT + metrics)
```

---

## DSP Concepts

| Concept | Formula / Detail |
|---|---|
| Nyquist frequency | f_N = f_s / 2 |
| Properly sampled | f_signal < f_N |
| Aliasing | f_signal ≥ f_N; apparent freq = fold of f_signal about f_N |
| FFT bin width | Δf = f_s / N |
| Spectral centroid | Σ(f · P(f)) / Σ P(f) |
| SNR | 10 · log₁₀(P_signal / P_noise) [dB] |

---

## ML Methodology

| Item | Detail |
|---|---|
| Algorithm | RandomForestClassifier (scikit-learn) |
| Features | 13 (mean, std, RMS, peak-peak, ZCR, dominant freq, spectral centroid/BW, LF/HF energy, power) |
| Training data | 2400 synthetic samples (400 per class × 6 classes) |
| Train/test split | 80% / 20% stratified |
| Reproducibility | Random seed = 42 |
| Model file | `models/signal_classifier.joblib` |

---

## Project Structure

```
SigNexis/
├── app.py                      # Streamlit application (8 pages)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── signal_dataset.csv      # Auto-generated training dataset
│
├── models/
│   ├── signal_classifier.joblib   # Trained Random Forest model
│   ├── feature_metadata.json      # Canonical feature names & order
│   └── evaluation_results.json    # Accuracy, confusion matrix, feature importance
│
├── dsp/
│   ├── __init__.py
│   ├── signal_generator.py    # 6-class signal + noise synthesis
│   ├── sampling.py            # Nyquist analysis, aliasing detection
│   ├── fft_analysis.py        # scipy.fft wrapper + spectral features
│   ├── features.py            # Unified 13-feature extraction pipeline
│   ├── filters.py             # LPF / HPF / BPF / Notch + recommendation
│   └── quality.py             # RMS, SNR, quality metrics
│
├── ml/
│   ├── __init__.py
│   ├── dataset.py             # Synthetic dataset generator
│   ├── train.py               # RF training + evaluation + saving
│   └── predict.py             # Inference function
│
├── audio/
│   ├── __init__.py
│   └── wav_processor.py       # WAV file ingestion & preprocessing
│
├── utils/
│   ├── __init__.py
│   └── validation.py          # Input validation helpers
│
└── tests/
    ├── test_sampling.py
    ├── test_fft.py
    ├── test_features.py
    ├── test_filters.py
    └── test_ml.py
```

---

## Installation

```bash
# Clone / open project
cd SigNexis

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 – Generate Training Dataset

```bash
python -m ml.dataset
```

Generates `data/signal_dataset.csv` (2400 rows) and `models/feature_metadata.json`.

### Step 2 – Train the Model

```bash
python -m ml.train
```

Trains the Random Forest, prints accuracy + classification report, saves:
- `models/signal_classifier.joblib`
- `models/evaluation_results.json`

### Step 3 – Launch the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Deploy on Streamlit Community Cloud

Use these exact settings in Streamlit Community Cloud:

- Repository: `Harikrishnan120A/SigNexis`
- Branch: `main`
- Main file path: `app.py`

This repository now includes:
- `runtime.txt` (`python-3.11`) for consistent cloud Python runtime
- `.streamlit/config.toml` for server defaults

Required runtime artifacts:
- `dsp/`, `ml/`, and `audio/` source packages
- `models/signal_classifier.joblib`
- `models/evaluation_results.json`

If required modules or model artifacts are missing at deployment time, the app exits gracefully with a clear error message instead of crashing.

### Step 4 – Run Tests

```bash
pytest
```

Or with verbose output:

```bash
pytest -v
```

---

## Dashboard Pages

| Page | Content |
|---|---|
| Dashboard | Live KPI cards, waveform, FFT spectrum |
| Signal Generator | Configure signal type, frequency, noise; generate |
| Sampling & Aliasing | Interactive Nyquist demo; aliasing visualisation |
| FFT Analysis | Spectral comparison (clean vs noisy) |
| ML Classification | Features, predictions, confidence, confusion matrix |
| Intelligent Filtering | Recommendation, manual override, filtered waveform |
| Before vs After | Side-by-side waveforms, spectra, quality metrics |
| About / WAV Upload | WAV file ingestion + full DSP→ML pipeline |

---

## Limitations

1. **Synthetic training data** – classifier performance on real-world audio is exploratory.
2. **SNR without reference** – true SNR requires a clean reference; heuristic estimates are labelled as such.
3. **Notch filter @ 50 Hz** – requires sampling rate > 100 Hz.
4. **No deep learning** – Random Forest chosen for interpretability and explainability (feature importance).

---

## Future Scope

- Real-world labelled signal dataset (EEG, ECG, seismic data)
- STFT / spectrogram for non-stationary signals
- Adaptive Least Mean Squares (LMS) filter
- Edge deployment on microcontroller with fixed-point DSP
- Uncertainty quantification in ML predictions

---

*SigNexis – Built for academic demonstration. All accuracy figures are from synthetic-data experiments.*
