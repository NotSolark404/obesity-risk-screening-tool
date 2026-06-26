"""
tests/test_engineer.py
Tests for src/features/engineer.py — add_features()
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import add_features

# Fixtures


@pytest.fixture
def sample_df():
    """Minimal encoded DataFrame matching the state AFTER binary/gender encoding
    but BEFORE feature engineering (as called inside ObesityPreprocessor)."""
    return pd.DataFrame(
        {
            "Gender": [1, 0, 1],  # Male=1, Female=0
            "Age": [25.0, 32.0, 45.0],
            "Height": [1.75, 1.60, 1.80],
            "Weight": [70.0, 55.0, 90.0],
            "family_history_with_overweight": [1, 0, 1],  # already encoded
            "FAVC": [1, 0, 1],
            "FCVC": [2.0, 3.0, 2.0],
            "NCP": [3.0, 2.0, 3.0],
            "CAEC": [1, 0, 2],
            "SMOKE": [0, 0, 1],
            "CH2O": [2.0, 1.0, 3.0],
            "SCC": [0, 1, 0],
            "FAF": [1.0, 0.0, 2.0],
            "TUE": [0.0, 1.0, 0.5],
            "CALC": [1, 0, 2],
            "MTRANS": ["Public_Transportation", "Walking", "Automobile"],
        }
    )


@pytest.fixture
def single_row_df():
    """Single-row DataFrame for deterministic value checks."""
    return pd.DataFrame(
        {
            "Gender": [1],
            "Age": [30.0],
            "Height": [1.80],
            "Weight": [80.0],
            "family_history_with_overweight": [1],
            "FAVC": [1],
            "FCVC": [2.0],
            "NCP": [3.0],
            "CAEC": [1],
            "SMOKE": [0],
            "CH2O": [2.0],
            "SCC": [0],
            "FAF": [1.0],
            "TUE": [0.0],
            "CALC": [0],
            "MTRANS": ["Public_Transportation"],
        }
    )


# 1. Output columns


class TestOutputColumns:

    EXPECTED_NEW_COLS = [
        "BMI",
        "height_gender_interaction",
        "age_weight_interaction",
        "weight_height_ratio",
        "family_history_weight_risk",
    ]

    def test_all_new_columns_present(self, sample_df):
        result = add_features(sample_df)
        for col in self.EXPECTED_NEW_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_df):
        result = add_features(sample_df)
        for col in sample_df.columns:
            assert col in result.columns, f"Original column dropped: {col}"

    def test_column_count_increases_by_five(self, sample_df):
        result = add_features(sample_df)
        assert result.shape[1] == sample_df.shape[1] + 5

    def test_row_count_unchanged(self, sample_df):
        result = add_features(sample_df)
        assert result.shape[0] == sample_df.shape[0]


# 2. BMI calculation


class TestBMI:

    def test_bmi_formula(self, single_row_df):
        """BMI = Weight / Height^2"""
        result = add_features(single_row_df)
        expected = 80.0 / (1.80**2)
        assert result["BMI"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_bmi_all_positive(self, sample_df):
        result = add_features(sample_df)
        assert (result["BMI"] > 0).all()

    def test_bmi_reasonable_range(self, sample_df):
        """BMI should be between 10 and 60 for realistic inputs."""
        result = add_features(sample_df)
        assert result["BMI"].between(10, 60).all()

    def test_bmi_vectorised_correctly(self, sample_df):
        result = add_features(sample_df)
        expected = sample_df["Weight"] / (sample_df["Height"] ** 2)
        pd.testing.assert_series_equal(
            result["BMI"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


#  3. height_gender_interaction


class TestHeightGenderInteraction:

    def test_formula(self, single_row_df):
        """height_gender_interaction = Height * Gender"""
        result = add_features(single_row_df)
        expected = 1.80 * 1  # Height * Gender(Male=1)
        assert result["height_gender_interaction"].iloc[0] == pytest.approx(expected)

    def test_zero_for_female(self, sample_df):
        """Female (Gender=0) should give interaction of 0."""
        result = add_features(sample_df)
        female_mask = sample_df["Gender"] == 0
        assert (result.loc[female_mask, "height_gender_interaction"] == 0).all()

    def test_equals_height_for_male(self, sample_df):
        """Male (Gender=1) should give interaction equal to Height."""
        result = add_features(sample_df)
        male_mask = sample_df["Gender"] == 1
        pd.testing.assert_series_equal(
            result.loc[male_mask, "height_gender_interaction"].reset_index(drop=True),
            sample_df.loc[male_mask, "Height"].reset_index(drop=True),
            check_names=False,
        )


#  4. age_weight_interaction


class TestAgeWeightInteraction:

    def test_formula(self, single_row_df):
        """age_weight_interaction = Age * Weight"""
        result = add_features(single_row_df)
        expected = 30.0 * 80.0
        assert result["age_weight_interaction"].iloc[0] == pytest.approx(expected)

    def test_all_positive(self, sample_df):
        result = add_features(sample_df)
        assert (result["age_weight_interaction"] > 0).all()

    def test_vectorised_correctly(self, sample_df):
        result = add_features(sample_df)
        expected = sample_df["Age"] * sample_df["Weight"]
        pd.testing.assert_series_equal(
            result["age_weight_interaction"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# 5. weight_height_ratio


class TestWeightHeightRatio:

    def test_formula(self, single_row_df):
        """weight_height_ratio = Weight / Height"""
        result = add_features(single_row_df)
        expected = 80.0 / 1.80
        assert result["weight_height_ratio"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_all_positive(self, sample_df):
        result = add_features(sample_df)
        assert (result["weight_height_ratio"] > 0).all()

    def test_vectorised_correctly(self, sample_df):
        result = add_features(sample_df)
        expected = sample_df["Weight"] / sample_df["Height"]
        pd.testing.assert_series_equal(
            result["weight_height_ratio"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# 6. family_history_weight_risk


class TestFamilyHistoryWeightRisk:

    def test_formula(self, single_row_df):
        """family_history_weight_risk = family_history_with_overweight * Weight"""
        result = add_features(single_row_df)
        expected = 1 * 80.0
        assert result["family_history_weight_risk"].iloc[0] == pytest.approx(expected)

    def test_zero_when_no_family_history(self, sample_df):
        result = add_features(sample_df)
        no_history_mask = sample_df["family_history_with_overweight"] == 0
        assert (result.loc[no_history_mask, "family_history_weight_risk"] == 0).all()

    def test_equals_weight_when_family_history(self, sample_df):
        result = add_features(sample_df)
        history_mask = sample_df["family_history_with_overweight"] == 1
        pd.testing.assert_series_equal(
            result.loc[history_mask, "family_history_weight_risk"].reset_index(drop=True),
            sample_df.loc[history_mask, "Weight"].reset_index(drop=True),
            check_names=False,
        )


# 7. No mutation of input


class TestImmutability:

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        original_shape = sample_df.shape
        _ = add_features(sample_df)
        assert list(sample_df.columns) == original_cols
        assert sample_df.shape == original_shape

    def test_returns_new_dataframe(self, sample_df):
        result = add_features(sample_df)
        assert result is not sample_df


#  8. No NaN values introduced


class TestNoNulls:

    def test_no_nulls_in_new_columns(self, sample_df):
        result = add_features(sample_df)
        new_cols = [
            "BMI",
            "height_gender_interaction",
            "age_weight_interaction",
            "weight_height_ratio",
            "family_history_weight_risk",
        ]
        assert result[new_cols].isnull().sum().sum() == 0

    def test_no_new_nulls_introduced(self, sample_df):
        result = add_features(sample_df)
        assert result.isnull().sum().sum() == 0


#  9. Data types


class TestDtypes:

    def test_bmi_is_float(self, sample_df):
        result = add_features(sample_df)
        assert result["BMI"].dtype == np.float64

    def test_all_new_cols_are_numeric(self, sample_df):
        result = add_features(sample_df)
        new_cols = [
            "BMI",
            "height_gender_interaction",
            "age_weight_interaction",
            "weight_height_ratio",
            "family_history_weight_risk",
        ]
        for col in new_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} should be numeric, got {result[col].dtype}"
