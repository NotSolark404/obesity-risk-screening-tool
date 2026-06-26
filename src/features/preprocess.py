import json
import pickle

from sklearn.model_selection import train_test_split

from src.config import (
    DATA_TRAIN,
    DATA_TEST,
    DATA_REF,
    PIPELINE_PKL,
    FEATURE_COLS,
    load_params,
)
from src.data.ingest import clean, load_raw
from src.features.obesity_preprocessor import ObesityPreprocessor


def main():
    params = load_params()
    raw = clean(load_raw())
    X, y = raw.drop(columns=["NObeyesdad"]), raw["NObeyesdad"]

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

    DATA_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    DATA_REF.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_PKL.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLS.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(DATA_TRAIN, index=False)
    test_df.to_csv(DATA_TEST, index=False)

    # Save all transformed features — not just BMI — so monitor can
    # drift-check any column, not only BMI
    ref = preprocessor.transform(X_train).copy()
    ref["NObeyesdad"] = y_train.values
    ref.to_csv(DATA_REF, index=False)

    with open(PIPELINE_PKL, "wb") as f:
        pickle.dump(preprocessor, f)
    with open(FEATURE_COLS, "w") as f:
        json.dump(preprocessor.feature_columns, f)

    print("[preprocess] train:", train_df.shape, "| test:", test_df.shape)
    print("[preprocess] reference:", ref.shape)
    print("[preprocess] pipeline.pkl saved")


if __name__ == "__main__":
    main()
