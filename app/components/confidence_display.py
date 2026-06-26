"""
Reusable result-display widgets: headline prediction, confidence meter,
and a sorted per-class probability breakdown.
"""

import pandas as pd
import streamlit as st

from components.feature_explanation import CLASS_INFO, class_color, class_label


# Prediction result widget function
# After user gets their results
def prediction_result(prediction: str, confidence: float, class_probs: dict) -> None:
    color = class_color(prediction)
    label = class_label(prediction)
    desc = CLASS_INFO.get(prediction, {}).get("desc", "")

    # Predicted Obesity Category
    st.markdown(
        f"""
        <div style="border-left:6px solid {color};
                    background:rgba(148,163,184,0.08);
                    padding:1rem 1.25rem; border-radius:8px; margin-bottom:1rem;">
            <div style="font-size:0.85rem; text-transform:uppercase;
                        letter-spacing:0.05em; opacity:0.7;">Predicted category</div>
            <div style="font-size:1.8rem; font-weight:700; color:{color};">{label}</div>
            <div style="opacity:0.8;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Confidence
    st.metric("Model confidence", f"{confidence * 100:.1f}%")
    st.progress(min(max(confidence, 0.0), 1.0))

    # Low Confidence Warning
    if confidence < 0.6:
        st.warning(
            "Low confidence: the model is uncertain between several " "categories. Treat this result as an estimate."
        )


# Probability Breakdown Widget
def probability_breakdown(class_probs: dict) -> None:
    st.markdown("##### Probability across all categories")

    rows = []
    for raw, prob in class_probs.items():
        rows.append(
            {
                "Category": class_label(raw),
                "Probability": float(prob),
                "_order": CLASS_INFO.get(raw, {}).get("order", 99),
            }
        )
    df = pd.DataFrame(rows).sort_values("_order").drop(columns="_order").set_index("Category")

    st.bar_chart(df, height=260)
    st.dataframe(
        df.style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
    )
