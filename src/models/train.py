import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_TRAIN, MODEL_PKL, load_params


def train():
    params = load_params()
    df = pd.read_csv(DATA_TRAIN)
    X_train = df.drop(columns=["NObeyesdad"])
    y_train = df["NObeyesdad"]

    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        min_samples_split=params["model"]["min_samples_split"],
        min_samples_leaf=params["model"]["min_samples_leaf"],
        random_state=params["data"]["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)

    print("[train] RandomForest trained and saved")


if __name__ == "__main__":
    train()
