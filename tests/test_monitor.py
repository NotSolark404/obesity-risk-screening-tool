"""
Tests for src/monitoring/monitor.py — check_data_drift()
No artifacts needed. Creates its own reference and production CSV files.
"""

import json
import os

import pandas as pd
import pytest

from src.config import DATA_REF, DRIFT_REPORT, PROD_LOG


@pytest.fixture(autouse=True)
def setup_dirs():
    # Runs before every test — creates all required directories
    DATA_REF.parent.mkdir(parents=True, exist_ok=True)
    PROD_LOG.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT.parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def reference_data():
    # Input: 200 rows of BMI values around 22-29 (normal weight range)
    # Written to data/reference/training_reference.csv
    df = pd.DataFrame(
        {
            "BMI": [22.0, 25.0, 28.0, 24.0, 26.0, 23.0, 27.0, 21.0, 29.0, 25.5] * 20,
            "NObeyesdad": ["Normal_Weight"] * 200,
        }
    )
    df.to_csv(DATA_REF, index=False)
    return df


@pytest.fixture
def similar_prod(reference_data):
    # Input: 200 rows of BMI values around 21-29 (similar to reference)
    # Written to monitoring/logs/production_prediction_log.csv
    df = pd.DataFrame(
        {
            "BMI": [22.5, 25.5, 27.5, 24.5, 26.5, 23.5, 27.5, 21.5, 29.5, 25.0] * 20,
            "prediction": ["Normal_Weight"] * 200,
        }
    )
    df.to_csv(PROD_LOG, index=False)
    return df


@pytest.fixture
def drifted_prod(reference_data):
    # Input: 200 rows of BMI values around 38-46 (obese range — completely different)
    # Written to monitoring/logs/production_prediction_log.csv
    df = pd.DataFrame(
        {
            "BMI": [40.0, 42.0, 45.0, 38.0, 41.0, 43.0, 39.0, 44.0, 46.0, 40.5] * 20,
            "prediction": ["Obesity_Type_III"] * 200,
        }
    )
    df.to_csv(PROD_LOG, index=False)
    return df


def test_no_drift_detected(reference_data, similar_prod):
    # How: KS-test on reference BMI (22-29) vs production BMI (21-29)
    # Input: overlapping distributions → p-value high → drift_detected=False
    from src.monitoring.monitor import check_data_drift

    assert check_data_drift()["drift_detected"] is False


def test_drift_detected(reference_data, drifted_prod):
    # How: KS-test on reference BMI (22-29) vs production BMI (38-46)
    # Input: completely separated distributions → p-value near 0 → drift_detected=True
    from src.monitoring.monitor import check_data_drift

    assert check_data_drift()["drift_detected"] is True


def test_result_has_p_value(reference_data, similar_prod):
    # How: checks return dict contains "p_value" key
    # Input: similar distributions — key must always be present regardless of result
    from src.monitoring.monitor import check_data_drift

    result = check_data_drift()
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_result_has_ks_stat(reference_data, similar_prod):
    # How: checks return dict contains "ks_stat" key with non-negative value
    # Input: similar distributions — KS statistic must always be present
    from src.monitoring.monitor import check_data_drift

    result = check_data_drift()
    assert "ks_stat" in result
    assert result["ks_stat"] >= 0.0


def test_no_prod_log_returns_no_drift(reference_data):
    # How: deletes production log if it exists, calls check_data_drift()
    # Input: no production log file — function must return gracefully
    from src.monitoring.monitor import check_data_drift

    if PROD_LOG.exists():
        os.remove(PROD_LOG)
    result = check_data_drift()
    assert result["drift_detected"] is False
    assert "reason" in result


def test_drift_report_written(reference_data, similar_prod):
    # How: deletes existing report, runs check, confirms file was created
    # Input: similar distributions — report must be written regardless of drift result
    from src.monitoring.monitor import check_data_drift

    if DRIFT_REPORT.exists():
        os.remove(DRIFT_REPORT)
    check_data_drift()
    assert DRIFT_REPORT.exists()
    with open(DRIFT_REPORT) as f:
        report = json.load(f)
    assert "drift_detected" in report


def test_p_value_range(reference_data, similar_prod):
    # How: runs check and asserts p_value is between 0 and 1
    # Input: similar distributions — p-value is always a valid probability
    from src.monitoring.monitor import check_data_drift

    result = check_data_drift()
    assert 0.0 <= result["p_value"] <= 1.0
