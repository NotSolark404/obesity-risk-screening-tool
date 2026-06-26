"""
tests/test_mlp.py

Owned by: Minh (Data & Model Specialist)
Focus: MLP-specific artifacts, evaluate_mlp pipeline outputs, version_log
       correctness for MLPClassifier entries, and predict.py MLP code path.

These tests validate the MLP as a deployed artifact (loading from .pkl,
reading evaluation_metrics_mlp.json) rather than training it from scratch.
Training-level MLP tests live in test_model.py.

Requires: `dvc repro` to have been run so that model_mlp.pkl and
          evaluation_metrics_mlp.json exist on disk.
"""

import csv
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier

from src.config import (
    DATA_TEST,
    EVAL_METRICS,
    MLP_EVAL_METRICS,
    MLP_PKL,
    PIPELINE_PKL,
    VERSION_LOG,
    load_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def mlp_model():
    """Load the trained MLP from the artifact produced by dvc repro."""
    pytest.importorskip("sklearn")  # guard — skip gracefully if sklearn missing
    if not MLP_PKL.exists():
        pytest.skip(f"model_mlp.pkl not found at {MLP_PKL} — run dvc repro first")
    with open(MLP_PKL, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def mlp_metrics():
    """Load the MLP evaluation metrics produced by evaluate_mlp."""
    if not MLP_EVAL_METRICS.exists():
        pytest.skip("evaluation_metrics_mlp.json not found — run dvc repro first")
    with open(MLP_EVAL_METRICS) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def preprocessor():
    """Load the fitted ObesityPreprocessor from the artifact on disk."""
    if not PIPELINE_PKL.exists():
        pytest.skip(f"pipeline.pkl not found at {PIPELINE_PKL} — run dvc repro first")
    with open(PIPELINE_PKL, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def test_data():
    """Load the preprocessed test set from disk."""
    if not DATA_TEST.exists():
        pytest.skip("test_preprocessed.csv not found — run dvc repro first")
    return pd.read_csv(DATA_TEST)


# ---------------------------------------------------------------------------
# TestMLPArtifact — model file exists and is the correct type
# ---------------------------------------------------------------------------


class TestMLPArtifact:

    def test_mlp_pkl_exists(self):
        assert MLP_PKL.exists(), f"model_mlp.pkl missing at {MLP_PKL}"

    def test_mlp_is_mlp_classifier(self, mlp_model):
        assert isinstance(mlp_model, MLPClassifier)

    def test_mlp_has_seven_classes(self, mlp_model):
        assert hasattr(mlp_model, "classes_")
        assert len(mlp_model.classes_) == 7

    def test_mlp_hidden_layer_sizes_match_params(self, mlp_model, params):
        expected = tuple(params["mlp"]["hidden_layer_sizes"])
        assert (
            mlp_model.hidden_layer_sizes == expected
        ), f"Expected hidden_layer_sizes {expected}, got {mlp_model.hidden_layer_sizes}"

    def test_mlp_activation_matches_params(self, mlp_model, params):
        assert mlp_model.activation == params["mlp"]["activation"]

    def test_mlp_pkl_round_trip(self, mlp_model, test_data, tmp_path):
        # Pickle and reload — predictions must be bit-identical.
        X_test = test_data.drop(columns=["NObeyesdad"])
        pkl_path = tmp_path / "mlp_roundtrip.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(mlp_model, f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, MLPClassifier)
        np.testing.assert_array_equal(mlp_model.predict(X_test), loaded.predict(X_test))


# ---------------------------------------------------------------------------
# TestMLPPredictions — model produces valid outputs on the real test set
# ---------------------------------------------------------------------------


class TestMLPPredictions:

    def test_mlp_predicts_correct_row_count(self, mlp_model, test_data):
        X_test = test_data.drop(columns=["NObeyesdad"])
        preds = mlp_model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_mlp_predictions_are_valid_classes(self, mlp_model, test_data):
        X_test = test_data.drop(columns=["NObeyesdad"])
        preds = mlp_model.predict(X_test)
        valid = set(mlp_model.classes_)
        assert all(p in valid for p in preds), "MLP produced a prediction outside known classes"

    def test_mlp_predict_proba_shape(self, mlp_model, test_data):
        X_test = test_data.drop(columns=["NObeyesdad"])
        proba = mlp_model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 7)

    def test_mlp_predict_proba_sums_to_one(self, mlp_model, test_data):
        X_test = test_data.drop(columns=["NObeyesdad"])
        proba = mlp_model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_mlp_single_row_inference(self, mlp_model, preprocessor):
        # Simulates the exact code path used in src/models/predict.py for model='mlp'.
        raw = pd.DataFrame(
            [
                {
                    "Gender": "Female",
                    "Age": 28.0,
                    "Height": 1.65,
                    "Weight": 90.0,
                    "family_history_with_overweight": "yes",
                    "FAVC": "yes",
                    "FCVC": 2.0,
                    "NCP": 3.0,
                    "CAEC": "Frequently",
                    "SMOKE": "no",
                    "CH2O": 2.0,
                    "SCC": "no",
                    "FAF": 0.0,
                    "TUE": 1.0,
                    "CALC": "Sometimes",
                    "MTRANS": "Automobile",
                }
            ]
        )
        X = preprocessor.transform(raw)
        prediction = mlp_model.predict(X)
        proba = mlp_model.predict_proba(X)
        assert len(prediction) == 1
        assert prediction[0] in mlp_model.classes_
        assert abs(proba[0].sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestMLPMetrics — evaluation_metrics_mlp.json content and threshold checks
# ---------------------------------------------------------------------------


class TestMLPMetrics:

    def test_mlp_metrics_file_exists(self):
        assert MLP_EVAL_METRICS.exists(), f"evaluation_metrics_mlp.json missing at {MLP_EVAL_METRICS}"

    def test_mlp_metrics_has_required_keys(self, mlp_metrics):
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in mlp_metrics, f"Missing metric key: {key}"

    def test_mlp_metrics_values_are_floats(self, mlp_metrics):
        for key in ("accuracy", "precision", "recall", "f1"):
            assert isinstance(mlp_metrics[key], float), f"{key} is not a float"

    def test_mlp_metrics_values_in_range(self, mlp_metrics):
        for key in ("accuracy", "precision", "recall", "f1"):
            val = mlp_metrics[key]
            assert 0.0 <= val <= 1.0, f"{key} value {val} is outside [0, 1]"

    def test_mlp_accuracy_above_threshold(self, mlp_metrics, params):
        # Threshold is params["mlp"]["performance_threshold"], not params["model"].
        threshold = params["mlp"]["performance_threshold"]
        acc = mlp_metrics["accuracy"]
        assert acc >= threshold, f"MLP accuracy {acc:.4f} below threshold {threshold}"

    def test_mlp_f1_above_threshold(self, mlp_metrics, params):
        # F1 should be in the same ballpark as accuracy for a balanced dataset.
        threshold = params["mlp"]["performance_threshold"]
        f1 = mlp_metrics["f1"]
        assert f1 >= threshold, f"MLP F1 {f1:.4f} below threshold {threshold}"

    def test_mlp_metrics_lower_than_rf(self):
        # RF is the primary model — its metrics should be >= MLP on this dataset.
        if not EVAL_METRICS.exists() or not MLP_EVAL_METRICS.exists():
            pytest.skip("Both metric files required for comparison")
        with open(EVAL_METRICS) as f:
            rf = json.load(f)
        with open(MLP_EVAL_METRICS) as f:
            mlp = json.load(f)
        assert (
            rf["accuracy"] >= mlp["accuracy"]
        ), f"RF accuracy {rf['accuracy']} should be >= MLP accuracy {mlp['accuracy']}"


# ---------------------------------------------------------------------------
# TestVersionLog — log_version() produces a valid MLPClassifier entry
# ---------------------------------------------------------------------------


class TestVersionLog:

    def test_version_log_records_mlp_entry(self):
        # If the version log exists, at least one MLPClassifier entry must be present
        # after a successful dvc repro run.
        if not VERSION_LOG.exists():
            pytest.skip("model_version_log.csv not found — run dvc repro first")
        with open(VERSION_LOG) as f:
            rows = list(csv.DictReader(f))
        mlp_rows = [r for r in rows if r.get("model_type") == "MLPClassifier"]
        assert len(mlp_rows) >= 1, "No MLPClassifier entry found in model_version_log.csv"

    def test_version_log_mlp_entry_has_valid_accuracy(self):
        if not VERSION_LOG.exists():
            pytest.skip("model_version_log.csv not found — run dvc repro first")
        with open(VERSION_LOG) as f:
            rows = list(csv.DictReader(f))
        mlp_rows = [r for r in rows if r.get("model_type") == "MLPClassifier"]
        if not mlp_rows:
            pytest.skip("No MLPClassifier entry to validate")
        acc = float(mlp_rows[-1]["accuracy"])
        assert 0.0 < acc <= 1.0, f"Invalid accuracy in version log: {acc}"

    def test_version_log_version_string_parseable(self):
        # Validates that the version string bug from version_log.py is fixed
        # and all entries have a correctly formatted vMAJOR.MINOR string.
        if not VERSION_LOG.exists():
            pytest.skip("model_version_log.csv not found — run dvc repro first")
        with open(VERSION_LOG) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            ver = row.get("version", "")
            assert ver.startswith("v"), f"Version '{ver}' does not start with 'v'"
            parts = ver.lstrip("v").split(".")
            assert len(parts) == 2, f"Version '{ver}' is not in vMAJOR.MINOR format"
            assert parts[1].isdigit(), f"Minor version in '{ver}' is not an integer"


# ---------------------------------------------------------------------------
# TestPredictMLPCodePath — validates src/models/predict.py for model='mlp'
# ---------------------------------------------------------------------------


class TestPredictMLPCodePath:

    def test_predict_function_mlp_path(self):
        # Import and call predict() with model='mlp' to ensure the MLP code
        # path in predict.py works end-to-end with the real artifacts on disk.
        if not MLP_PKL.exists() or not PIPELINE_PKL.exists():
            pytest.skip("Artifacts not found — run dvc repro first")
        from src.models.predict import predict

        sample = {
            "Gender": "Male",
            "Age": 22.0,
            "Height": 1.78,
            "Weight": 75.0,
            "family_history_with_overweight": "no",
            "FAVC": "no",
            "FCVC": 3.0,
            "NCP": 3.0,
            "CAEC": "Sometimes",
            "SMOKE": "no",
            "CH2O": 3.0,
            "SCC": "no",
            "FAF": 2.0,
            "TUE": 0.0,
            "CALC": "no",
            "MTRANS": "Walking",
        }
        prediction, confidence, class_probs = predict(sample, model="mlp")
        assert isinstance(prediction, str)
        assert 0.0 < confidence <= 1.0
        assert len(class_probs) == 7
        assert abs(sum(class_probs.values()) - 1.0) < 1e-4
