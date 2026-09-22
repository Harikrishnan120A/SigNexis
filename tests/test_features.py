"""tests/test_features.py – Pytest tests for the feature extraction pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from dsp.features import (
    extract_features,
    features_to_vector,
    FEATURE_NAMES,
    NUM_FEATURES,
)


class TestFeatureExtraction:
    def _sine(self, freq=1000, fs=8000, duration=0.5):
        t = np.arange(0, duration, 1.0 / fs)
        return np.sin(2 * np.pi * freq * t), fs

    def test_returns_all_feature_names(self):
        signal, fs = self._sine()
        features = extract_features(signal, fs)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_correct_feature_count(self):
        signal, fs = self._sine()
        features = extract_features(signal, fs)
        assert len(features) == NUM_FEATURES

    def test_no_nan_values(self):
        signal, fs = self._sine()
        features = extract_features(signal, fs)
        for name, val in features.items():
            assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"

    def test_no_nan_on_noisy_signal(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.5, 4000)
        features = extract_features(signal, 8000)
        for name, val in features.items():
            assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"

    def test_rms_correct(self):
        """RMS of sin(2πft) should be ≈ 1/√2 ≈ 0.707."""
        t = np.arange(0, 1.0, 1.0 / 8000)
        signal = np.sin(2 * np.pi * 1000 * t)
        features = extract_features(signal, 8000)
        assert features["rms"] == pytest.approx(1.0 / np.sqrt(2), abs=0.01)

    def test_features_to_vector_length(self):
        signal, fs = self._sine()
        features = extract_features(signal, fs)
        vec = features_to_vector(features)
        assert len(vec) == NUM_FEATURES

    def test_features_to_vector_order(self):
        signal, fs = self._sine()
        features = extract_features(signal, fs)
        vec = features_to_vector(features)
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == pytest.approx(features[name])

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            extract_features(np.array([]), 8000)

    def test_peak_to_peak_correct(self):
        signal = np.array([-1.0, 0.0, 1.0])
        features = extract_features(signal, 8000)
        assert features["peak_to_peak"] == pytest.approx(2.0)

    def test_zero_run_counts_as_one_crossing(self):
        signal = np.array([1.0, 0.0, -1.0])
        features = extract_features(signal, 8000)
        assert features["zero_crossing_rate"] == pytest.approx(0.5)
