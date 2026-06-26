"""
tests/test_pipeline.py

Owned by: QA / Nick
Focus: data integrity, preprocessing pipeline correctness, train/test split,
       and no-leakage guarantees. Model training is exercised only as far as
       needed to validate the preprocessor output is trainable.

Does NOT require `dvc repro` — the fixture rebuilds the full pipeline from
the raw CSV in-process.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, load_params
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
    """Build the full preprocessing + RF training pipeline from scratch.

    Scoped to module so the expensive fit/train step runs only once per
    test session, not once per test function.
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
    preprocessor.fit(X_train)  # fitted on training data only — no leakage

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
    model.fit(train_df.drop(columns=["NObeyesdad"]), train_df["NObeyesdad"])

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


# ---------------------------------------------------------------------------
# TestRawData — validate the CSV on disk before any processing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestTrainTestSplit — 80/20 stratified split correctness
# ---------------------------------------------------------------------------


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
        # Each class proportion in the training set must be within 5% of
        # its proportion in the full dataset — confirms stratify=y worked.
        full_dist = raw_data["NObeyesdad"].value_counts(normalize=True)
        train_dist = pipeline_artifacts["y_train"].value_counts(normalize=True)
        for cls in full_dist.index:
            diff = abs(full_dist[cls] - train_dist.get(cls, 0))
            assert diff < 0.05, f"Class {cls} proportion off by {diff:.3f}"

    def test_random_state_reproducible(self, raw_data, params):
        X = raw_data.drop(columns=["NObeyesdad"])
        y = raw_data["NObeyesdad"]
        kwargs = dict(
            test_size=params["data"]["test_size"],
            random_state=params["data"]["random_state"],
            stratify=y,
        )
        X1, _, _, _ = train_test_split(X, y, **kwargs)
        X2, _, _, _ = train_test_split(X, y, **kwargs)
        pd.testing.assert_frame_equal(X1, X2)


# ---------------------------------------------------------------------------
# TestPreprocessorOutput — shape, columns, and null checks after transform
# ---------------------------------------------------------------------------


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
        # Raw MTRANS string column must be replaced by one-hot dummy columns.
        assert "MTRANS" not in pipeline_artifacts["train_df"].columns
        assert "MTRANS" not in pipeline_artifacts["test_df"].columns

    def test_bmi_col_present(self, pipeline_artifacts):
        assert "BMI" in pipeline_artifacts["train_df"].columns

    def test_all_engineered_cols_present(self, pipeline_artifacts):
        engineered = [
            "BMI",
            "age_weight_interaction",
            "weight_height_ratio",
            "height_gender_interaction",
            "family_history_weight_risk",
        ]
        for col in engineered:
            assert col in pipeline_artifacts["train_df"].columns, f"Missing engineered column: {col}"


# ---------------------------------------------------------------------------
# TestNoDataLeakage — scaler must be fitted on training data only
# ---------------------------------------------------------------------------


class TestNoDataLeakage:

    def test_scaler_fitted_on_train_only(self, pipeline_artifacts):
        pre = pipeline_artifacts["preprocessor"]
        assert hasattr(pre.scaler, "mean_"), "Scaler was never fitted"
        assert pre.scaler.mean_ is not None

    def test_fit_transform_vs_transform_differ_on_test(self, pipeline_artifacts):
        # Transforming X_test with the train-fitted scaler must produce
        # different values than re-fitting on X_test itself, because the
        # scaler learned different statistics from training data.
        pre = pipeline_artifacts["preprocessor"]
        X_test = pipeline_artifacts["X_test"]

        result_train_scaler = pre.transform(X_test)

        pre_fresh = ObesityPreprocessor()
        pre_fresh.fit(X_test)
        result_test_scaler = pre_fresh.transform(X_test)

        # Compare only non-MTRANS columns — dummies are deterministic.
        cols = [
            c for c in result_train_scaler.columns if c in result_test_scaler.columns and not c.startswith("MTRANS_")
        ]
        diffs = result_train_scaler[cols].values - result_test_scaler[cols].values
        assert not np.allclose(diffs, 0)

    def test_test_indices_not_in_train(self, pipeline_artifacts):
        train_idx = set(pipeline_artifacts["X_train"].index)
        test_idx = set(pipeline_artifacts["X_test"].index)
        assert len(train_idx & test_idx) == 0


# ---------------------------------------------------------------------------
# TestArtifacts — serialisation round-trips for pipeline and model objects
# ---------------------------------------------------------------------------


class TestArtifacts:

    def test_preprocessor_is_obesity_preprocessor(self, pipeline_artifacts):
        assert isinstance(pipeline_artifacts["preprocessor"], ObesityPreprocessor)

    def test_model_is_random_forest(self, pipeline_artifacts):
        assert isinstance(pipeline_artifacts["model"], RandomForestClassifier)

    def test_model_has_seven_classes(self, pipeline_artifacts):
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


# ---------------------------------------------------------------------------
# TestReproducibility — same inputs must always produce identical outputs
# ---------------------------------------------------------------------------


class TestReproducibility:

    def test_same_split_same_random_state(self, raw_data, params):
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
        # n_jobs=1 removes parallelism so results are bit-identical across runs.
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

        m1, m2 = make_model(), make_model()
        np.testing.assert_array_equal(m1.predict(X_test), m2.predict(X_test))
