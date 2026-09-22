"""
ml/predict.py
=============
Model inference module for SigNexis.

Provides a clean predict_signal() function that:
    1. Extracts canonical features from the signal
    2. Loads the trained RandomForest model (cached)
    3. Returns predicted class + per-class probabilities
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import joblib

from dsp.features import extract_features, features_to_vector, FEATURE_NAMES

MODEL_PATH = ROOT / "models" / "signal_classifier.joblib"

# Module-level cache so the model is loaded once per process
_model_cache: dict = {}


def load_model(model_path: Path = MODEL_PATH):
    """Load and cache the trained RandomForest model.

    Raises
    ------
    FileNotFoundError if the model file doesn't exist.
    """
    key = str(model_path)
    if key not in _model_cache:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Train the model first:\n"
                "    python -m ml.dataset\n"
                "    python -m ml.train"
            )
        _model_cache[key] = joblib.load(model_path)
    return _model_cache[key]


def _validate_model_contract(model, model_path: Path) -> None:
    """Reject models whose persisted input shape differs from this pipeline."""
    expected = len(FEATURE_NAMES)
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and actual != expected:
        raise ValueError(
            f"Model expects {actual} features, but this pipeline provides {expected}."
        )

    metadata_path = Path(model_path).parent / "feature_metadata.json"
    if metadata_path.exists():
        import json

        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("feature_names") != FEATURE_NAMES:
            raise ValueError(
                f"Feature metadata does not match the active pipeline: {metadata_path}"
            )


@dataclass
class Prediction:
    """Container for ML prediction output."""

    predicted_class: str
    confidence: float                  # probability of predicted class
    class_probabilities: Dict[str, float]
    features_used: Dict[str, float]
    model_path: str


def predict_signal(
    signal: np.ndarray,
    sampling_rate: float,
    model_path: Path = MODEL_PATH,
) -> Prediction:
    """Predict noise/condition class for *signal*.

    Parameters
    ----------
    signal        : 1-D float array – the signal to classify
    sampling_rate : sample rate of *signal* in Hz
    model_path    : path to saved .joblib model

    Returns
    -------
    Prediction dataclass
    """
    if len(signal) == 0:
        raise ValueError("Cannot predict on empty signal.")
    if not np.all(np.isfinite(signal)):
        # Replace non-finite values but warn
        signal = np.where(np.isfinite(signal), signal, 0.0)

    # ── Feature extraction (same pipeline as training) ─────────────────────────
    features = extract_features(signal, sampling_rate)
    X = features_to_vector(features).reshape(1, -1)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model(model_path)
    _validate_model_contract(model, Path(model_path))

    # ── Predict ────────────────────────────────────────────────────────────────
    predicted_class = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    class_probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    confidence = float(class_probabilities[predicted_class])

    return Prediction(
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        features_used=features,
        model_path=str(model_path),
    )
