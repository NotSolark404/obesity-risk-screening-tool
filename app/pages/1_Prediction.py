"""Prediction page — collect inputs, run the model, display results."""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path so src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Pickle fix — required to load pipeline.pkl
from src.features.preprocess import ObesityPreprocessor  # noqa: E402

sys.modules["__main__"].ObesityPreprocessor = ObesityPreprocessor

from src.models.predict import predict  # noqa: E402

# Calling pages library
from components.confidence_display import prediction_result, probability_breakdown  # noqa: E402
from components.feature_explanation import obesity_form  # noqa: E402

# Prediction Page Config
st.set_page_config(page_title="Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Predict obesity category")
st.caption("Fill in the details below and click **Predict**.")

# Model selector
model_choice = st.sidebar.radio(
    "Choose model",
    options=["Random Forest", "MLP"],
    help="Random Forest is the primary model. MLP is a neural network alternative.",
)
model_key = "rf" if model_choice == "Random Forest" else "mlp"

# Map display name to the model_type string logged in production_prediction_log.csv
MODEL_TYPE_LOG = {
    "Random Forest": "RandomForestClassifier",
    "MLP": "MLPClassifier",
}

st.sidebar.caption(
    "**🌲 Random Forest** *(recommended)*: the primary model. "
    "Selected for its strong generalisation and interpretable confidence scores.\n\n"
    "**🧠 MLP**: a two-layer neural network alternative. "
    "Available for comparison — use Random Forest for clinical screening."
)

# Prediction Form
with st.form("prediction_form"):
    inputs = obesity_form()
    submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

if submitted:
    try:
        prediction, confidence, class_probs = predict(inputs, model=model_key)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # Log prediction to production log (used by Monitoring page)
    log_path = "monitoring/logs/production_prediction_log.csv"
    os.makedirs("monitoring/logs", exist_ok=True)
    row = dict(inputs)
    row["BMI"] = round(inputs["Weight"] / (inputs["Height"] ** 2), 4)
    row["model_type"] = MODEL_TYPE_LOG[model_choice]  # log full classifier name
    row["prediction"] = prediction
    row["confidence"] = round(confidence, 4)
    pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))

    # Display prediction results
    st.divider()
    st.caption(f"Result from: **{model_choice}**")
    left, right = st.columns([1, 1])
    with left:
        prediction_result(prediction, confidence, class_probs)
    with right:
        probability_breakdown(class_probs)

    # Check the user input submitted
    with st.expander("See the raw input sent to the model"):
        st.json(inputs)
