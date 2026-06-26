import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import DATA_TEST, EVAL_METRICS, MODEL_PKL, load_params
from src.models.version_log import log_version


def evaluate():
    params = load_params()
    df = pd.read_csv(DATA_TEST)
    X_test = df.drop(columns=["NObeyesdad"])
    y_test = df["NObeyesdad"]

    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    threshold = params["model"]["performance_threshold"]
    if metrics["accuracy"] < threshold:
        raise ValueError(f"[evaluate] accuracy {metrics['accuracy']} below threshold {threshold}")

    EVAL_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    log_version(model_type="RandomForestClassifier", metrics_path=EVAL_METRICS)
    print("[evaluate]", metrics)


if __name__ == "__main__":
    evaluate()
