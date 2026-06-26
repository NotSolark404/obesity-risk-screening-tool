import pandas as pd
from src.config import DATA_RAW


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    print(f"[ingest] {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[ingest] dropped {before - len(df)} duplicates")
    return df
