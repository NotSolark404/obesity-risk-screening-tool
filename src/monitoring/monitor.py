import json

import pandas as pd
from scipy import stats

from src.config import DATA_REF, DRIFT_REPORT, PROD_LOG, load_params


def check_data_drift(col: str = "BMI") -> dict:
    params = load_params()
    threshold = params["monitoring"]["drift_threshold"]

    ref = pd.read_csv(DATA_REF)

    if not PROD_LOG.exists():
        print("[monitor] no production log yet")
        return {"drift_detected": False, "reason": "no production log yet"}

    prod = pd.read_csv(PROD_LOG)

    if col not in prod.columns:
        return {"drift_detected": False, "reason": f"{col} not in production log"}

    stat, p = stats.ks_2samp(ref[col], prod[col])
    result = {
        "drift_detected": bool(p < threshold),
        "p_value": round(float(p), 6),
        "ks_stat": round(float(stat), 6),
    }

    DRIFT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT, "w") as f:
        json.dump(result, f, indent=2)

    print("[monitor]", result)
    return result


if __name__ == "__main__":
    check_data_drift()
