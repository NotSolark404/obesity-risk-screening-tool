import json
import pickle

import pandas as pd

from src.config import FEATURE_COLS, MLP_PKL, MODEL_PKL, PIPELINE_PKL


def load_artifacts(model: str = "rf"):
    """Load pipeline, model and feature columns. model='rf' or 'mlp'."""
    with open(PIPELINE_PKL, "rb") as f:
        pipeline = pickle.load(f)
    model_path = MODEL_PKL if model == "rf" else MLP_PKL
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(FEATURE_COLS) as f:
        feature_columns = json.load(f)
    return pipeline, clf, feature_columns


def predict(input_dict: dict, model: str = "rf") -> tuple:
    """Predict obesity category. model='rf' (Random Forest) or 'mlp' (MLP)."""
    pipeline, clf, feature_columns = load_artifacts(model)
    df = pd.DataFrame([input_dict])
    X = pipeline.transform(df).reindex(columns=feature_columns, fill_value=0)
    prediction = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = float(proba.max())
    class_probs = dict(zip(clf.classes_, [round(float(p), 4) for p in proba]))
    return prediction, confidence, class_probs
