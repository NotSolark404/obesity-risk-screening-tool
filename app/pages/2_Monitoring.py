"""Monitoring page — model version log, live data-drift check, and predictions."""

import json
import os

import pandas as pd
import streamlit as st

# Monitoring Page Config
st.set_page_config(page_title="Monitoring", page_icon="📈", layout="wide")
st.title("📈 Monitoring Dashboard")

# ── Model Version Log ────────────────────────────────────────────────────────
st.subheader("Model Version Log")
log_path = "artifacts/metrics/model_version_log.csv"
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    if not df.empty:
        # Show latest RF and MLP entries separately
        rf_rows = df[df["model_type"] == "RandomForestClassifier"]
        mlp_rows = df[df["model_type"] == "MLPClassifier"]
        latest_rf = rf_rows.iloc[-1] if not rf_rows.empty else None
        latest_mlp = mlp_rows.iloc[-1] if not mlp_rows.empty else None

        c1, c2, c3, c4 = st.columns(4)
        if latest_rf is not None:
            c1.metric("🌲 RF Version", latest_rf["version"])
            c2.metric("RF Trained", latest_rf["datetime"])
        if latest_mlp is not None:
            c3.metric("🧠 MLP Version", latest_mlp["version"])
            c4.metric("MLP Trained", latest_mlp["datetime"])

    st.dataframe(df, use_container_width=True)
else:
    st.info("No version log yet.")

st.divider()

# ── Model Comparison (RF vs MLP) ─────────────────────────────────────────────
st.subheader("Model Comparison — Random Forest vs MLP")

rf_path = "artifacts/metrics/evaluation_metrics.json"
mlp_path = "artifacts/metrics/evaluation_metrics_mlp.json"

rf_metrics = json.load(open(rf_path)) if os.path.exists(rf_path) else None
mlp_metrics = json.load(open(mlp_path)) if os.path.exists(mlp_path) else None

if rf_metrics or mlp_metrics:
    col_rf, col_mlp = st.columns(2)

    with col_rf:
        st.markdown("##### 🌲 Random Forest")
        if rf_metrics:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{rf_metrics['accuracy'] * 100:.2f}%")
            c2.metric("F1 Score", f"{rf_metrics['f1'] * 100:.2f}%")
            c1.metric("Precision", f"{rf_metrics['precision'] * 100:.2f}%")
            c2.metric("Recall", f"{rf_metrics['recall'] * 100:.2f}%")
        else:
            st.info("Run the evaluate stage to generate RF metrics.")

    with col_mlp:
        st.markdown("##### 🧠 MLP")
        if mlp_metrics:
            c1, c2 = st.columns(2)
            c1.metric(
                "Accuracy",
                f"{mlp_metrics['accuracy'] * 100:.2f}%",
                delta=f"{(mlp_metrics['accuracy'] - rf_metrics['accuracy']) * 100:.2f}%" if rf_metrics else None,
            )
            c2.metric(
                "F1 Score",
                f"{mlp_metrics['f1'] * 100:.2f}%",
                delta=f"{(mlp_metrics['f1'] - rf_metrics['f1']) * 100:.2f}%" if rf_metrics else None,
            )
            c1.metric("Precision", f"{mlp_metrics['precision'] * 100:.2f}%")
            c2.metric("Recall", f"{mlp_metrics['recall'] * 100:.2f}%")
        else:
            st.info("Run python -m src.models.train_mlp then evaluate_mlp to generate MLP metrics.")
else:
    st.info("No model metrics found yet.")

st.divider()

# ── Data Drift ────────────────────────────────────────────────────────────────
st.subheader("Data Drift (KS-test on BMI)")
drift_path = "monitoring/alerts/drift_report.json"
if os.path.exists(drift_path):
    with open(drift_path) as f:
        drift = json.load(f)
    label = "DRIFT DETECTED" if drift["drift_detected"] else "No Drift"
    st.metric("BMI Drift", label, delta=f"p = {drift.get('p_value', 'N/A')}")
else:
    st.info("No drift report yet. Run monitor.py to generate.")

st.divider()

# ── Recent Predictions ────────────────────────────────────────────────────────
st.subheader("Recent Predictions")
pred_path = "monitoring/logs/production_prediction_log.csv"
if os.path.exists(pred_path):
    try:
        pred_df = pd.read_csv(pred_path, on_bad_lines="skip")
        st.metric("Total Predictions", len(pred_df))

        # Show breakdown by model_type (full classifier name)
        if "model_type" in pred_df.columns:
            counts = pred_df["model_type"].value_counts()
            c1, c2 = st.columns(2)
            c1.metric("🌲 RandomForestClassifier", counts.get("RandomForestClassifier", 0))
            c2.metric("🧠 MLPClassifier", counts.get("MLPClassifier", 0))

        # Display last 20 predictions, prioritising the most useful columns
        display_cols = ["model_type", "prediction", "confidence", "BMI", "Gender", "Age", "Weight", "Height"]
        available_cols = [c for c in display_cols if c in pred_df.columns]
        remaining_cols = [c for c in pred_df.columns if c not in available_cols]
        st.dataframe(pred_df[available_cols + remaining_cols].tail(20), use_container_width=True)

    except Exception as e:
        st.error(f"Could not read prediction log: {e}")
        if st.button("Clear corrupted log"):
            os.remove(pred_path)
            st.rerun()
else:
    st.info("No predictions logged yet.")
