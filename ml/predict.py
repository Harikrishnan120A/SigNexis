import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
from dsp.features import extract_features, features_to_vector
ROOT=Path(__file__).resolve().parent.parent
MODEL_PATH=ROOT/"models"/"signal_classifier.joblib"
_cache={}
def load_model(model_path=MODEL_PATH):
    key=str(model_path)
    if key not in _cache:
        if not Path(model_path).exists(): raise FileNotFoundError(f"Model file not found: {model_path}")
        _cache[key]=joblib.load(model_path)
    return _cache[key]
@dataclass
class Prediction:
    predicted_class: str; confidence: float; class_probabilities: dict; features_used: dict; model_path: str
def predict_signal(signal, sampling_rate, model_path=MODEL_PATH):
    signal=np.where(np.isfinite(signal),signal,0.0); features=extract_features(signal,sampling_rate); model=load_model(model_path); x=features_to_vector(features).reshape(1,-1); label=str(model.predict(x)[0]); probabilities={str(c):float(p) for c,p in zip(model.classes_,model.predict_proba(x)[0])}
    return Prediction(label,probabilities[label],probabilities,features,str(model_path))
