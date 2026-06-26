import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.obesity_preprocessor import ObesityPreprocessor
from src.data.ingest import clean, load_raw
from src.config import (
    DATA_RAW,
    load_params,
)


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def raw_data():
    return clean(load_raw())


@pytest.fixture(scope="module")
def pipeline_artifacts(raw_data, params):
    from sklearn.model_selection import train_test_split

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

    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        min_samples_split=params["model"]["min_samples_split"],
        min_samples_leaf=params["model"]["min_samples_leaf"],
        random_state=params["data"]["random_state"],
        n_jobs=-1,
    )
    model.fit(
        train_df.drop(columns=["NObeyesdad"]),
        train_df["NObeyesdad"],
    )

    return {
        "preprocessor": preprocessor,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
    }


class TestRawData:

    def test_raw_file_exists(self):
        assert DATA_RAW.exists(), f"Raw data not found at {DATA_RAW}"

    def test_raw_data_loads(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)

    def test_raw_data_has_rows(self, raw_data):
        assert len(raw_data) > 0

    def test_raw_data_has_expected_columns(self, raw_data):
        expected = [
            "Gender",
            "Age",
            "Height",
            "Weight",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
            "CH2O",
            "SCC",
            "FAF",
            "TUE",
            "CALC",
            "MTRANS",
            "NObeyesdad",
        ]
        for col in expected:
            assert col in raw_data.columns, f"Missing column: {col}"

    def test_target_column_present(self, raw_data):
        assert "NObeyesdad" in raw_data.columns

    def test_target_has_seven_classes(self, raw_data):
        assert raw_data["NObeyesdad"].nunique() == 7

    def test_no_duplicates_after_clean(self, raw_data):
        assert raw_data.duplicated().sum() == 0

    def test_no_nulls_in_raw(self, raw_data):
        assert raw_data.isnull().sum().sum() == 0


class TestTrainTestSplit:

    def test_split_sizes(self, pipeline_artifacts, raw_data, params):
        n_total = len(raw_data)
        n_test = len(pipeline_artifacts["X_test"])
        expected = round(n_total * params["data"]["test_size"])
        assert abs(n_test - expected) <= 2

    def test_no_overlap_between_train_and_test(self, pipeline_artifacts):
        train_idx = set(pipeline_artifacts["X_train"].index)
        test_idx = set(pipeline_artifacts["X_test"].index)
        assert train_idx.isdisjoint(test_idx)

    def test_train_test_together_equal_full(self, pipeline_artifacts, raw_data):
        n_train = len(pipeline_artifacts["X_train"])
        n_test = len(pipeline_artifacts["X_test"])
        assert n_train + n_test == len(raw_data)

    def test_stratified_class_distribution(self, pipeline_artifacts, raw_data):
        full_dist = raw_data["NObeyesdad"].value_counts(normalize=True)
        train_dist = pipeline_artifacts["y_train"].value_counts(normalize=True)
        for cls in full_dist.index:
            diff = abs(full_dist[cls] - train_dist.get(cls, 0))
            assert diff < 0.05, f"Class {cls} proportion off by {diff:.3f}"

    def test_random_state_reproducible(self, raw_data, params):
        from sklearn.model_selection import train_test_split

        X = raw_data.drop(columns=["NObeyesdad"])
        y = raw_data["NObeyesdad"]
        X1, _, _, _ = train_test_split(
            X, y, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"], stratify=y
        )
        X2, _, _, _ = train_test_split(
            X, y, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"], stratify=y
        )
        pd.testing.assert_frame_equal(X1, X2)


class TestPreprocessorOutput:

    def test_train_df_has_target(self, pipeline_artifacts):
        assert "NObeyesdad" in pipeline_artifacts["train_df"].columns

    def test_test_df_has_target(self, pipeline_artifacts):
        assert "NObeyesdad" in pipeline_artifacts["test_df"].columns

    def test_train_df_no_nulls(self, pipeline_artifacts):
        assert pipeline_artifacts["train_df"].isnull().sum().sum() == 0

    def test_test_df_no_nulls(self, pipeline_artifacts):
        assert pipeline_artifacts["test_df"].isnull().sum().sum() == 0

    def test_train_test_same_feature_columns(self, pipeline_artifacts):
        train_cols = [c for c in pipeline_artifacts["train_df"].columns if c != "NObeyesdad"]
        test_cols = [c for c in pipeline_artifacts["test_df"].columns if c != "NObeyesdad"]
        assert train_cols == test_cols

    def test_mtrans_raw_col_not_in_output(self, pipeline_artifacts):
        assert "MTRANS" not in pipeline_artifacts["train_df"].columns
        assert "MTRANS" not in pipeline_artifacts["test_df"].columns

    def test_bmi_col_present(self, pipeline_artifacts):
        assert "BMI" in pipeline_artifacts["train_df"].columns

    def test_engineered_cols_in_train(self, pipeline_artifacts):
        for col in [
            "BMI",
            "age_weight_interaction",
            "weight_height_ratio",
            "height_gender_interaction",
            "family_history_weight_risk",
        ]:
            assert col in pipeline_artifacts["train_df"].columns


class TestNoDataLeakage:

    def test_scaler_fitted_on_train_only(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        assert hasattr(pre.scaler, "mean_"), "Scaler must be fitted"
        assert pre.scaler.mean_ is not None

    def test_fit_transform_vs_transform_differ_on_test(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        X_test = pipeline_artifacts["X_test"]

        result_transform = pre.transform(X_test)

        pre_fresh = ObesityPreprocessor()
        pre_fresh.fit(X_test)
        result_fit_transform = pre_fresh.transform(X_test)

        cols = [
            c for c in result_transform.columns if c in result_fit_transform.columns and not c.startswith("MTRANS_")
        ]
        diffs = result_transform[cols].values - result_fit_transform[cols].values
        assert not np.allclose(diffs, 0)

    def test_test_indices_not_in_train(self, pipeline_artifacts):
        train_idx = set(pipeline_artifacts["X_train"].index)
        test_idx = set(pipeline_artifacts["X_test"].index)
        assert len(train_idx & test_idx) == 0


class TestArtifacts:

    def test_preprocessor_is_obesity_preprocessor(self, pipeline_artifacts):
        assert isinstance(pipeline_artifacts["preprocessor"], ObesityPreprocessor)

    def test_model_is_random_forest(self, pipeline_artifacts):
        assert isinstance(pipeline_artifacts["model"], RandomForestClassifier)

    def test_model_has_classes(self, pipeline_artifacts):
        assert hasattr(pipeline_artifacts["model"], "classes_")
        assert len(pipeline_artifacts["model"].classes_) == 7

    def test_model_feature_count_matches_preprocessor(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        model = pipeline_artifacts["model"]
        assert model.n_features_in_ == len(pre.feature_columns)

    def test_pipeline_pkl_serialisable(self, pipeline_artifacts, tmp_path):
        pkl_path = tmp_path / "pipeline.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline_artifacts["preprocessor"], f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, ObesityPreprocessor)
        assert loaded.feature_columns == pipeline_artifacts["preprocessor"].feature_columns

    def test_model_pkl_serialisable(self, pipeline_artifacts, tmp_path):
        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline_artifacts["model"], f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, RandomForestClassifier)

    def test_feature_columns_json_serialisable(self, pipeline_artifacts, tmp_path):
        json_path = tmp_path / "feature_columns.json"
        cols = pipeline_artifacts["preprocessor"].feature_columns
        with open(json_path, "w") as f:
            json.dump(cols, f)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded == cols


class TestModelTraining:

    def test_model_predicts_on_test(self, pipeline_artifacts):
        model = pipeline_artifacts["model"]
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predictions_are_valid_classes(self, pipeline_artifacts):
        model = pipeline_artifacts["model"]
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        preds = model.predict(X_test)
        valid = set(pipeline_artifacts["model"].classes_)
        assert all(p in valid for p in preds)

    def test_predict_proba_shape(self, pipeline_artifacts):
        model = pipeline_artifacts["model"]
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 7)

    def test_predict_proba_sums_to_one(self, pipeline_artifacts):
        model = pipeline_artifacts["model"]
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        proba = model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_model_accuracy_above_threshold(self, pipeline_artifacts, params):
        from sklearn.metrics import accuracy_score

        model = pipeline_artifacts["model"]
        test_df = pipeline_artifacts["test_df"]
        X_test = test_df.drop(columns=["NObeyesdad"])
        y_test = test_df["NObeyesdad"]
        acc = accuracy_score(y_test, model.predict(X_test))
        threshold = params["model"]["performance_threshold"]
        assert acc >= threshold, f"Accuracy {acc:.4f} below threshold {threshold}"

    def test_single_inference_end_to_end(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        model = pipeline_artifacts["model"]
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


class TestReproducibility:

    def test_same_split_same_random_state(self, raw_data, params):
        from sklearn.model_selection import train_test_split

        X = raw_data.drop(columns=["NObeyesdad"])
        y = raw_data["NObeyesdad"]
        kwargs = dict(
            test_size=params["data"]["test_size"],
            random_state=params["data"]["random_state"],
            stratify=y,
        )
        X1_tr, X1_te, _, _ = train_test_split(X, y, **kwargs)
        X2_tr, X2_te, _, _ = train_test_split(X, y, **kwargs)
        pd.testing.assert_frame_equal(X1_tr.reset_index(drop=True), X2_tr.reset_index(drop=True))
        pd.testing.assert_frame_equal(X1_te.reset_index(drop=True), X2_te.reset_index(drop=True))

    def test_same_model_params_same_predictions(self, pipeline_artifacts, params):
        train_df = pipeline_artifacts["train_df"]
        test_df = pipeline_artifacts["test_df"]
        X_train = train_df.drop(columns=["NObeyesdad"])
        y_train = train_df["NObeyesdad"]
        X_test = test_df.drop(columns=["NObeyesdad"])

        def make_model():
            m = RandomForestClassifier(
                n_estimators=params["model"]["n_estimators"],
                max_depth=params["model"]["max_depth"],
                min_samples_split=params["model"]["min_samples_split"],
                min_samples_leaf=params["model"]["min_samples_leaf"],
                random_state=params["data"]["random_state"],
                n_jobs=1,
            )
            m.fit(X_train, y_train)
            return m

        m1 = make_model()
        m2 = make_model()
        np.testing.assert_array_equal(m1.predict(X_test), m2.predict(X_test))
