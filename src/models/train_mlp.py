import pickle

import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.config import DATA_TRAIN, MLP_PKL, load_params


def train_mlp():
    params = load_params()["mlp"]
    df = pd.read_csv(DATA_TRAIN)
    X_train = df.drop(columns=["NObeyesdad"])
    y_train = df["NObeyesdad"]

    model = MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        max_iter=params["max_iter"],
        random_state=params["random_state"],
        learning_rate_init=params["learning_rate_init"],
    )
    model.fit(X_train, y_train)

    MLP_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(MLP_PKL, "wb") as f:
        pickle.dump(model, f)

    print("[train_mlp] MLP trained and saved")


if __name__ == "__main__":
    train_mlp()
