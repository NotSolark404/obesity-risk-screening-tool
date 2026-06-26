from pathlib import Path
import yaml

# Project root — two levels up from src/config.py
ROOT = Path(__file__).resolve().parent.parent

# params
PARAMS_PATH = ROOT / "params.yaml"

# data
DATA_RAW = ROOT / "data" / "raw" / "ObesityDataSet_raw_and_data_sinthetic.csv"
DATA_TRAIN = ROOT / "data" / "processed" / "train_preprocessed.csv"
DATA_TEST = ROOT / "data" / "processed" / "test_preprocessed.csv"
DATA_REF = ROOT / "data" / "reference" / "training_reference.csv"
DATA_NEW = ROOT / "data" / "new_data"

# artifacts
PIPELINE_PKL = ROOT / "artifacts" / "models" / "pipeline.pkl"
MODEL_PKL = ROOT / "artifacts" / "models" / "model.pkl"
FEATURE_COLS = ROOT / "artifacts" / "preprocessing" / "feature_columns.json"
EVAL_METRICS = ROOT / "artifacts" / "metrics" / "evaluation_metrics.json"
VERSION_LOG = ROOT / "artifacts" / "metrics" / "model_version_log.csv"

# monitoring
PROD_LOG = ROOT / "monitoring" / "logs" / "production_prediction_log.csv"
DRIFT_REPORT = ROOT / "monitoring" / "alerts" / "drift_report.json"
ALERT_HISTORY = ROOT / "monitoring" / "alerts" / "alert_history.json"


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


# MLP artifacts
MLP_PKL = ROOT / "artifacts" / "models" / "model_mlp.pkl"
MLP_EVAL_METRICS = ROOT / "artifacts" / "metrics" / "evaluation_metrics_mlp.json"
