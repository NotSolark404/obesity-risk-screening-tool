import csv
import json
from datetime import datetime
from pathlib import Path

from src.config import EVAL_METRICS, VERSION_LOG

HEADERS = ["version", "datetime", "model_type", "accuracy", "f1_weighted"]


def _ensure_log_exists():
    VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not VERSION_LOG.exists():
        with open(VERSION_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        print("[version_log] created model_version_log.csv")


def _next_version(rows: list) -> str:
    """Derive next version string from the last row, with a safe fallback."""
    if not rows:
        return "v1.1"
    last = rows[-1].get("version", "v1.0")
    try:
        # strip leading 'v', split on '.', expect exactly major.minor
        parts = last.lstrip("v").split(".")
        if len(parts) == 2:
            major, minor = parts
            return f"v{major}.{int(minor) + 1}"
        else:
            # malformed — just bump a counter on the whole thing
            return f"v1.{len(rows) + 1}"
    except (ValueError, IndexError):
        return f"v1.{len(rows) + 1}"


def log_version(model_type: str = "RandomForestClassifier", metrics_path: Path = EVAL_METRICS):
    """Append a version entry. Pass model_type and metrics_path for MLP."""
    _ensure_log_exists()

    with open(metrics_path) as f:
        metrics = json.load(f)

    with open(VERSION_LOG) as f:
        rows = list(csv.DictReader(f))

    new_version = _next_version(rows)

    row = {
        "version": new_version,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_type": model_type,
        "accuracy": metrics["accuracy"],
        "f1_weighted": metrics["f1"],
    }

    with open(VERSION_LOG, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

    print(f"[version_log] logged {new_version} ({model_type}) — acc {metrics['accuracy']}")


if __name__ == "__main__":
    log_version()
