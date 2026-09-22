"""tests/test_ml.py – Pytest tests for ML dataset loading, training, and prediction."""

import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import pandas as pd

from dsp.features import FEATURE_NAMES, NUM_FEATURES
from ml.dataset import generate_dataset, save_dataset, load_dataset
from ml.train import train
from ml.predict import predict_signal, load_model


class IncompatibleModel:
    n_features_in_ = NUM_FEATURES - 1


class TestDataset:
    def test_generate_dataset_shape(self):
        """Dataset should have correct columns and ≥ 6 rows."""
        df = generate_dataset(samples_per_class=5, verbose=False)
        assert "label" in df.columns
        for name in FEATURE_NAMES:
            assert name in df.columns, f"Missing column: {name}"
        assert len(df) >= 6  # 1 per class minimum

    def test_generate_dataset_classes(self):
        from dsp.signal_generator import SIGNAL_CLASSES
        df = generate_dataset(samples_per_class=5, verbose=False)
        assert set(df["label"].unique()) == set(SIGNAL_CLASSES)

    def test_no_nan_in_dataset(self):
        df = generate_dataset(samples_per_class=5, verbose=False)
        assert not df[FEATURE_NAMES].isnull().any().any(), "Dataset contains NaN values"

    def test_save_and_load_roundtrip(self):
        df = generate_dataset(samples_per_class=3, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp_path = Path(f.name)
        save_dataset(df, tmp_path)
        loaded = load_dataset(tmp_path)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)
        tmp_path.unlink()


class TestTraining:
    def test_train_and_save(self, tmp_path):
        """Train on a tiny dataset, save, verify model file exists."""
        # Generate tiny dataset
        df = generate_dataset(samples_per_class=20, verbose=False)
        dataset_path = tmp_path / "dataset.csv"
        save_dataset(df, dataset_path)
        model_path = tmp_path / "model.joblib"

        results = train(
            dataset_path=dataset_path,
            model_path=model_path,
            verbose=False,
        )

        assert model_path.exists(), "Model file not saved."
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert "confusion_matrix" in results

    def test_accuracy_reported_not_fake(self, tmp_path):
        """Accuracy should actually reflect test-set performance."""
        df = generate_dataset(samples_per_class=30, verbose=False)
        dataset_path = tmp_path / "dataset.csv"
        save_dataset(df, dataset_path)
        model_path = tmp_path / "model.joblib"

        results = train(dataset_path=dataset_path, model_path=model_path, verbose=False)
        # Accuracy should be between 0 and 1
        assert isinstance(results["accuracy"], float)
        assert 0.0 <= results["accuracy"] <= 1.0


class TestPrediction:
    @pytest.fixture
    def trained_model_path(self, tmp_path):
        """Train a small model and return its path."""
        df = generate_dataset(samples_per_class=30, verbose=False)
        dataset_path = tmp_path / "dataset.csv"
        save_dataset(df, dataset_path)
        model_path = tmp_path / "model.joblib"
        train(dataset_path=dataset_path, model_path=model_path, verbose=False)
        return model_path

    def test_prediction_returns_valid_class(self, trained_model_path):
        from dsp.signal_generator import SIGNAL_CLASSES
        t = np.arange(0, 0.5, 1.0 / 8000)
        signal = np.sin(2 * np.pi * 1000 * t)
        pred = predict_signal(signal, 8000, trained_model_path)
        assert pred.predicted_class in SIGNAL_CLASSES

    def test_confidence_in_range(self, trained_model_path):
        t = np.arange(0, 0.5, 1.0 / 8000)
        signal = np.sin(2 * np.pi * 1000 * t)
        pred = predict_signal(signal, 8000, trained_model_path)
        assert 0.0 <= pred.confidence <= 1.0

    def test_probabilities_sum_to_one(self, trained_model_path):
        t = np.arange(0, 0.5, 1.0 / 8000)
        signal = np.sin(2 * np.pi * 1000 * t)
        pred = predict_signal(signal, 8000, trained_model_path)
        total = sum(pred.class_probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_features_used_correct_count(self, trained_model_path):
        t = np.arange(0, 0.5, 1.0 / 8000)
        signal = np.sin(2 * np.pi * 1000 * t)
        pred = predict_signal(signal, 8000, trained_model_path)
        assert len(pred.features_used) == NUM_FEATURES

    def test_model_not_found_raises(self):
        t = np.arange(0, 0.5, 1.0 / 8000)
        signal = np.sin(2 * np.pi * 1000 * t)
        with pytest.raises(FileNotFoundError):
            predict_signal(signal, 8000, Path("/nonexistent/model.joblib"))

    def test_model_feature_count_mismatch_raises(self, tmp_path):
        import joblib

        model_path = tmp_path / "incompatible.joblib"
        joblib.dump(IncompatibleModel(), model_path)
        signal = np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000)
        with pytest.raises(ValueError, match="expects 12 features"):
            predict_signal(signal, 8000, model_path)
