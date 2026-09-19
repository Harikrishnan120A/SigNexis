"""
ml/dataset.py
=============
Synthetic dataset generator for SigNexis.

Generates at least 2000 labelled samples (≥ 333 per class).
Each sample is a signal → DSP-feature row.

Run directly:
    python -m ml.dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── allow imports from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from dsp.signal_generator import generate_signal, SIGNAL_CLASSES
from dsp.features import extract_features, FEATURE_NAMES, save_feature_metadata

DATASET_PATH = ROOT / "data" / "signal_dataset.csv"
METADATA_PATH = ROOT / "models" / "feature_metadata.json"

# ── dataset config ─────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 400          # ×6 classes → 2400 total
RANDOM_SEED = 42
FS_OPTIONS = [4000, 8000, 16000, 22050]   # Hz
FREQ_OPTIONS = [100, 200, 500, 1000, 1500, 2000]  # Hz
NOISE_AMP_RANGE = (0.1, 0.8)
DURATION = 0.5                   # seconds – keep dataset generation fast


def _sample_params(rng: np.random.Generator) -> dict:
    """Random signal parameters for one sample."""
    fs = int(rng.choice(FS_OPTIONS))
    # Ensure signal_freq < Nyquist so the signal itself is valid
    max_freq = fs / 2.0 - 50
    freq_choices = [f for f in FREQ_OPTIONS if f < max_freq]
    if not freq_choices:
        freq_choices = [100]
    freq = float(rng.choice(freq_choices))
    amplitude = float(rng.uniform(0.5, 2.0))
    noise_amp = float(rng.uniform(*NOISE_AMP_RANGE))
    signal_type = str(rng.choice(["sine", "multi_sine"]))
    return dict(
        frequency=freq,
        amplitude=amplitude,
        sampling_rate=fs,
        noise_amplitude=noise_amp,
        signal_type=signal_type,
    )


def generate_dataset(
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate the full synthetic dataset.

    Returns
    -------
    DataFrame with columns = FEATURE_NAMES + ['label']
    """
    rng = np.random.default_rng(seed)
    rows = []

    for cls in SIGNAL_CLASSES:
        if verbose:
            print(f"  Generating {samples_per_class} samples for class: {cls}")

        for _ in range(samples_per_class):
            params = _sample_params(rng)

            _, _, noisy = generate_signal(
                signal_type=params["signal_type"],
                frequency=params["frequency"],
                amplitude=params["amplitude"],
                duration=DURATION,
                sampling_rate=params["sampling_rate"],
                noise_type=cls,
                noise_amplitude=params["noise_amplitude"],
                rng=rng,
            )

            features = extract_features(noisy, params["sampling_rate"])
            row = {name: features[name] for name in FEATURE_NAMES}
            row["label"] = cls
            rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    return df


def save_dataset(df: pd.DataFrame, path: Path = DATASET_PATH) -> None:
    """Save dataset CSV to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nDataset saved -> {path}  ({len(df)} rows)")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the dataset CSV from *path*."""
    return pd.read_csv(path)


if __name__ == "__main__":
    print("=" * 60)
    print("SigNexis - Dataset Generator")
    print("=" * 60)

    print(f"\nSaving feature metadata -> {METADATA_PATH}")
    save_feature_metadata(METADATA_PATH)

    print(f"\nGenerating dataset ({SAMPLES_PER_CLASS} samples/class x 6 classes)...")
    df = generate_dataset(verbose=True)
    save_dataset(df)

    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nFeatures : {FEATURE_NAMES}")
    print(f"Total samples : {len(df)}")
    print("\nDone.")
