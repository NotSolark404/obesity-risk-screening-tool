"""
Obesity Risk Classifier — Streamlit home page.

Run from the project root with:
    streamlit run app/Home.py
"""

import json
import os

import streamlit as st  # noqa: E402

# Landing Page Config
st.set_page_config(
    page_title="Obesity Risk Screening Tool",
    layout="wide",
)
st.title("Clinical Obesity Risk Screening Tool")
st.caption("Navigate using the sidebar.")
st.markdown(
    "A machine-learning tool that estimates a person's **weight / obesity "
    "category** from lifestyle and physical attributes, using a **Random Forest** "
    "model trained on the UCI Obesity dataset."
)

# Headline Metrics — Random Forest (primary model) only
metrics_path = "artifacts/metrics/evaluation_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    st.caption("🌲 **Primary model performance — Random Forest Classifier** (evaluated on held-out test set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
    c2.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
    c3.metric("Recall", f"{metrics.get('recall', 0) * 100:.1f}%")
    c4.metric("F1 score", f"{metrics.get('f1', 0) * 100:.1f}%")

st.divider()

# Info Section (How to use + category legend)
c1, c2 = st.columns(2)
with c1:
    st.subheader("How to use it")
    st.markdown(
        "1. Open the **Prediction** page from the sidebar.\n"
        "2. Enter the person's details, eating habits and lifestyle.\n"
        "3. Get a predicted category, a confidence score and a full "
        "probability breakdown.\n\n"
        "The **Monitoring** page shows model performance and version history."
    )

    st.subheader("Available models")
    st.markdown(
        "**🌲 Random Forest** *(primary — recommended for all predictions)*: "
        "an ensemble of decision trees trained on the UCI Obesity dataset. "
        "Selected as the primary model for its strong generalisation, "
        "interpretable feature importances, and reliable confidence scores.\n\n"
        "**🧠 MLP (Neural Network)** *(alternative)*: a two-layer neural network "
        "available as a secondary option on the Prediction page. Useful for "
        "comparison but the Random Forest should be preferred for clinical screening."
    )

with c2:
    st.subheader("The 7 categories")
    from components.feature_explanation import CLASS_INFO

    for data, info in sorted(CLASS_INFO.items(), key=lambda kv: kv[1]["order"]):
        st.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"border-radius:50%;background:{info['color']};margin-right:8px;'></span>"
            f"**{info['label']}** — {info['desc']}",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "⚠️ This tool is for educational/demonstration purposes only and is not a "
    "medical diagnosis. Consult a healthcare professional for health advice."
)
