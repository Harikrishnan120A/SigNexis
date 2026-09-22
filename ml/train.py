"""
ml/train.py
===========
Train the RandomForestClassifier for SigNexis.

Steps:
    1. Load data/signal_dataset.csv
    2. Stratified train/test split
    3. Train RandomForestClassifier
    4. Evaluate (accuracy, confusion matrix, classification report)
    5. Save model to models/signal_classifier.joblib
    6. Save feature metadata to models/feature_metadata.json

Run:
    python -m ml.train
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dsp.features import FEATURE_NAMES, save_feature_metadata
from ml.dataset import DATASET_PATH, METADATA_PATH, load_dataset

MODEL_PATH = ROOT / "models" / "signal_classifier.joblib"
EVAL_PATH = ROOT / "models" / "evaluation_results.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def train(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    eval_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train and save the RandomForest classifier.

    Parameters
    ----------
    dataset_path : Path to the CSV dataset.
    model_path   : Where to save the .joblib model file.
    eval_path    : Where to save evaluation_results.json.  Defaults to
                   <model_path.parent>/evaluation_results.json so that
                   test runs using tmp_path never overwrite the production
                   evaluation file.
    verbose      : Print progress.

    Returns
    -------
    dict with accuracy, classification_report, confusion_matrix, feature_importance
    """
    # Derive eval_path from model_path so tests using tmp_path stay isolated
    if eval_path is None:
        eval_path = Path(model_path).parent / "evaluation_results.json"

    # ── Load ─────────────────────────────────────────────────────────────────
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run `python -m ml.dataset` first."
        )
    df = load_dataset(dataset_path)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=str)

    if verbose:
        print(f"  Loaded {len(df)} samples, {len(FEATURE_NAMES)} features")
        print(f"  Classes: {list(np.unique(y))}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    if verbose:
        print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report_str = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_).tolist()
    classes = list(clf.classes_)

    feature_importance = {
        name: float(imp)
        for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)
    }

    if verbose:
        print(f"\n  Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n{report_str}")
        print("  Feature importances:")
        for name, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
            print(f"    {name:30s}: {imp:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    if verbose:
        print(f"\n  Model saved -> {model_path}")

    # ── Save feature metadata ─────────────────────────────────────────────────
    metadata_path = model_path.parent / "feature_metadata.json"
    save_feature_metadata(metadata_path)
    if verbose:
        print(f"  Feature metadata saved -> {metadata_path}")

    # ── Save evaluation results ───────────────────────────────────────────────
    eval_results = {
        "accuracy": accuracy,
        "test_size": TEST_SIZE,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": classes,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "feature_importance": feature_importance,
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
    }
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as fh:
        json.dump(eval_results, fh, indent=2)
    if verbose:
        print(f"  Evaluation results saved -> {eval_path}")

    return eval_results


if __name__ == "__main__":
    print("=" * 60)
    print("SigNexis - Model Training")
    print("=" * 60)
    results = train(verbose=True)
    print(f"\nFinal accuracy: {results['accuracy']*100:.2f}%")
    print("Done.")
