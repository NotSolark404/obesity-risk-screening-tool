import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.features.engineer import add_features

BINARY_COLS = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]

# Maps handle both lowercase (training data) and Title case (Streamlit form inputs)
BINARY_MAP = {"yes": 1, "no": 0, "Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0, "male": 1, "female": 0}
CAEC_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3, "No": 0}
CALC_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3, "No": 0}


class ObesityPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = StandardScaler()
        self.scale_cols = None
        self.mtrans_columns = None
        self.feature_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        df = self._encode_and_engineer(X)
        self.scale_cols = [c for c in df.columns if c != "MTRANS"]
        self.scaler.fit(df[self.scale_cols])
        self.mtrans_columns = list(pd.get_dummies(df["MTRANS"], prefix="MTRANS").columns)
        self.feature_columns = list(self._apply(df).columns)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = self._encode_and_engineer(X)
        result = self._apply(df).reindex(columns=self.feature_columns, fill_value=0)
        # Catch any residual NaN values — fill with 0 so MLP doesn't crash
        if result.isnull().any().any():
            result = result.fillna(0)
        return result

    def _encode_and_engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Gender"] = df["Gender"].map(GENDER_MAP)
        for col in BINARY_COLS:
            df[col] = df[col].map(BINARY_MAP)
        df["CAEC"] = df["CAEC"].map(CAEC_MAP)
        df["CALC"] = df["CALC"].map(CALC_MAP)
        return add_features(df)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.scale_cols] = self.scaler.transform(df[self.scale_cols])
        mtrans_dummies = pd.get_dummies(df["MTRANS"], prefix="MTRANS").reindex(
            columns=self.mtrans_columns, fill_value=0
        )
        return pd.concat([df.drop(columns=["MTRANS"]), mtrans_dummies], axis=1)
