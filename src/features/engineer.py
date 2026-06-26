import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["height_gender_interaction"] = df["Height"] * df["Gender"]
    df["age_weight_interaction"] = df["Age"] * df["Weight"]
    df["weight_height_ratio"] = df["Weight"] / df["Height"]
    df["family_history_weight_risk"] = df["family_history_with_overweight"] * df["Weight"]
    return df
