"""
tests/test_model.py

Owned by: Minh (Data & Model Specialist)
Focus: model prediction behaviour, accuracy thresholds, and end-to-end
       inference for both the Random Forest (primary) and MLP (secondary)
       models. Data/pipeline integrity tests live in test_pipeline.py.

Does NOT require `dvc repro` — the fixture rebuilds both models from the
raw CSV in-process using params.yaml for all hyperparameters.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.config import load_params
from src.data.ingest import clean, load_raw
from src.features.obesity_preprocessor import ObesityPreprocessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def raw_data():
    return clean(load_raw())


@pytest.fixture(scope="module")
def pipeline_artifacts(raw_data, params):
    """Build the preprocessor, RandomForest, and MLP from scratch.

    Both models share the same preprocessed train/test split so comparisons
    are fair. Scoped to module so training runs once per test session.
    """
    X = raw_data.drop(columns=["NObeyesdad"])
    y = raw_data["NObeyesdad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    preprocessor = ObesityPreprocessor()
    preprocessor.fit(X_train)

    train_df = preprocessor.transform(X_train).copy()
    train_df["NObeyesdad"] = y_train.values
    test_df = preprocessor.transform(X_test).copy()
    test_df["NObeyesdad"] = y_test.values

    rf_params = params["model"]
    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=params["data"]["random_state"],
        n_jobs=-1,
    )
    rf.fit(train_df.drop(columns=["NObeyesdad"]), train_df["NObeyesdad"])

    mlp_params = params["mlp"]
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(mlp_params["hidden_layer_sizes"]),
        activation=mlp_params["activation"],
        max_iter=mlp_params["max_iter"],
        random_state=mlp_params["random_state"],
        learning_rate_init=mlp_params["learning_rate_init"],
    )
    mlp.fit(train_df.drop(columns=["NObeyesdad"]), train_df["NObeyesdad"])

    return {
        "preprocessor": preprocessor,
        "rf": rf,
        "mlp": mlp,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
    }


# ---------------------------------------------------------------------------
# TestRandomForest — RF prediction behaviour and accuracy threshold
# ---------------------------------------------------------------------------


class TestRandomForest:

    def test_rf_predicts_correct_row_count(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        preds = pipeline_artifacts["rf"].predict(X_test)
        assert len(preds) == len(X_test)

    def test_rf_predictions_are_valid_classes(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        preds = pipeline_artifacts["rf"].predict(X_test)
        valid = set(pipeline_artifacts["rf"].classes_)
        assert all(p in valid for p in preds)

    def test_rf_predict_proba_shape(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        proba = pipeline_artifacts["rf"].predict_proba(X_test)
        assert proba.shape == (len(X_test), 7)

    def test_rf_predict_proba_sums_to_one(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        proba = pipeline_artifacts["rf"].predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_rf_accuracy_above_threshold(self, pipeline_artifacts, params):
        # Threshold comes from params["model"], not params["mlp"].
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        y_test = test_df["NObeyesdad"]
        acc = accuracy_score(y_test, pipeline_artifacts["rf"].predict(X_test))
        threshold = params["model"]["performance_threshold"]
        assert acc >= threshold, f"RF accuracy {acc:.4f} below threshold {threshold}"

    def test_rf_single_inference_end_to_end(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        model = pipeline_artifacts["rf"]
        raw = pd.DataFrame(
            [
                {
                    "Gender": "Male",
                    "Age": 25.0,
                    "Height": 1.75,
                    "Weight": 70.0,
                    "family_history_with_overweight": "yes",
                    "FAVC": "yes",
                    "FCVC": 2.0,
                    "NCP": 3.0,
                    "CAEC": "Sometimes",
                    "SMOKE": "no",
                    "CH2O": 2.0,
                    "SCC": "no",
                    "FAF": 1.0,
                    "TUE": 0.5,
                    "CALC": "no",
                    "MTRANS": "Public_Transportation",
                }
            ]
        )
        X = pre.transform(raw)
        prediction = model.predict(X)
        proba = model.predict_proba(X)
        assert len(prediction) == 1
        assert prediction[0] in model.classes_
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_rf_pkl_serialisable(self, pipeline_artifacts, tmp_path):
        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline_artifacts["rf"], f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, RandomForestClassifier)
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        np.testing.assert_array_equal(
            pipeline_artifacts["rf"].predict(X_test),
            loaded.predict(X_test),
        )


# ---------------------------------------------------------------------------
# TestMLP — MLP prediction behaviour and accuracy threshold
# ---------------------------------------------------------------------------


class TestMLP:

    def test_mlp_predicts_correct_row_count(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        preds = pipeline_artifacts["mlp"].predict(X_test)
        assert len(preds) == len(X_test)

    def test_mlp_predictions_are_valid_classes(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        preds = pipeline_artifacts["mlp"].predict(X_test)
        valid = set(pipeline_artifacts["mlp"].classes_)
        assert all(p in valid for p in preds)

    def test_mlp_has_seven_classes(self, pipeline_artifacts):
        assert len(pipeline_artifacts["mlp"].classes_) == 7

    def test_mlp_predict_proba_shape(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        proba = pipeline_artifacts["mlp"].predict_proba(X_test)
        assert proba.shape == (len(X_test), 7)

    def test_mlp_predict_proba_sums_to_one(self, pipeline_artifacts):
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        proba = pipeline_artifacts["mlp"].predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_mlp_accuracy_above_threshold(self, pipeline_artifacts, params):
        # Threshold comes from params["mlp"], not params["model"].
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        y_test = test_df["NObeyesdad"]
        acc = accuracy_score(y_test, pipeline_artifacts["mlp"].predict(X_test))
        threshold = params["mlp"]["performance_threshold"]
        assert acc >= threshold, f"MLP accuracy {acc:.4f} below threshold {threshold}"

    def test_mlp_feature_count_matches_preprocessor(self, pipeline_artifacts):
        # MLP must expect the same number of features the preprocessor outputs.
        pre = pipeline_artifacts["preprocessor"]
        mlp = pipeline_artifacts["mlp"]
        assert mlp.n_features_in_ == len(pre.feature_columns)

    def test_mlp_single_inference_end_to_end(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        model = pipeline_artifacts["mlp"]
        raw = pd.DataFrame(
            [
                {
                    "Gender": "Female",
                    "Age": 30.0,
                    "Height": 1.62,
                    "Weight": 85.0,
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
        X = pre.transform(raw)
        prediction = model.predict(X)
        proba = model.predict_proba(X)
        assert len(prediction) == 1
        assert prediction[0] in model.classes_
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_mlp_pkl_serialisable(self, pipeline_artifacts, tmp_path):
        pkl_path = tmp_path / "model_mlp.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline_artifacts["mlp"], f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, MLPClassifier)
        X_test = pipeline_artifacts["test_df"].drop(columns=["NObeyesdad"])
        np.testing.assert_array_equal(
            pipeline_artifacts["mlp"].predict(X_test),
            loaded.predict(X_test),
        )

    def test_mlp_hidden_layer_sizes_match_params(self, pipeline_artifacts, params):
        # Sanity check that the MLP was built with the sizes declared in params.yaml.
        expected = tuple(params["mlp"]["hidden_layer_sizes"])
        assert pipeline_artifacts["mlp"].hidden_layer_sizes == expected


# ---------------------------------------------------------------------------
# TestModelComparison — both models operate on identical feature space
# ---------------------------------------------------------------------------


class TestModelComparison:

    def test_both_models_predict_same_classes(self, pipeline_artifacts):
        # RF and MLP must agree on the set of possible output classes.
        rf_classes = set(pipeline_artifacts["rf"].classes_)
        mlp_classes = set(pipeline_artifacts["mlp"].classes_)
        assert rf_classes == mlp_classes

    def test_rf_accuracy_exceeds_mlp(self, pipeline_artifacts, params):
        # RF is the primary model and should outperform MLP on this dataset.
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        y_test = test_df["NObeyesdad"]
        rf_acc = accuracy_score(y_test, pipeline_artifacts["rf"].predict(X_test))
        mlp_acc = accuracy_score(y_test, pipeline_artifacts["mlp"].predict(X_test))
        assert rf_acc >= mlp_acc, f"RF ({rf_acc:.4f}) should outperform MLP ({mlp_acc:.4f}) on this dataset"

    def test_both_models_above_their_own_thresholds(self, pipeline_artifacts, params):
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        y_test = test_df["NObeyesdad"]
        rf_acc = accuracy_score(y_test, pipeline_artifacts["rf"].predict(X_test))
        mlp_acc = accuracy_score(y_test, pipeline_artifacts["mlp"].predict(X_test))
        assert rf_acc >= params["model"]["performance_threshold"]
        assert mlp_acc >= params["mlp"]["performance_threshold"]


# ---------------------------------------------------------------------------
# TestReproducibility — fixed random_state must produce identical predictions
# ---------------------------------------------------------------------------


class TestReproducibility:

    def test_same_rf_params_same_predictions(self, pipeline_artifacts, params):
        train_df = pipeline_artifacts["train_df"]
        test_df = pipeline_artifacts["test_df"]
        X_train = train_df.drop(columns=["NObeyesdad"])
        y_train = train_df["NObeyesdad"]
        X_test = test_df.drop(columns=["NObeyesdad"])

        def make_rf():
            m = RandomForestClassifier(
                n_estimators=params["model"]["n_estimators"],
                max_depth=params["model"]["max_depth"],
                min_samples_split=params["model"]["min_samples_split"],
                min_samples_leaf=params["model"]["min_samples_leaf"],
                random_state=params["data"]["random_state"],
                n_jobs=1,  # force single-thread for bit-identical output
            )
            m.fit(X_train, y_train)
            return m

        m1, m2 = make_rf(), make_rf()
        np.testing.assert_array_equal(m1.predict(X_test), m2.predict(X_test))

    def test_same_mlp_params_same_predictions(self, pipeline_artifacts, params):
        train_df = pipeline_artifacts["train_df"]
        test_df = pipeline_artifacts["test_df"]
        X_train = train_df.drop(columns=["NObeyesdad"])
        y_train = train_df["NObeyesdad"]
        X_test = test_df.drop(columns=["NObeyesdad"])

        def make_mlp():
            m = MLPClassifier(
                hidden_layer_sizes=tuple(params["mlp"]["hidden_layer_sizes"]),
                activation=params["mlp"]["activation"],
                max_iter=params["mlp"]["max_iter"],
                random_state=params["mlp"]["random_state"],
                learning_rate_init=params["mlp"]["learning_rate_init"],
            )
            m.fit(X_train, y_train)
            return m

        m1, m2 = make_mlp(), make_mlp()
        np.testing.assert_array_equal(m1.predict(X_test), m2.predict(X_test))
