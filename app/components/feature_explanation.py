"""
Feature metadata + input-form rendering + human-readable class info.

Centralises everything the UI needs to know about the 16 raw model inputs
and the 7 obesity classes, so the prediction page stays clean.
"""

import streamlit as st

# -------------------------------------------------------------------
# Input feature definitions (raw values, exactly as the training data)
# -------------------------------------------------------------------
# Numeric ranges are taken from the raw ObesityDataSet

# Dropdown options for binary and frequency fields
YES_NO = ["No", "Yes"]
FREQ = ["No", "Sometimes", "Frequently", "Always"]

NUMERIC_FEATURES = {
    "Age": dict(label="Age (years)", min=14, max=80, value=25, step=1, help="Age in years."),
    "Height": dict(label="Height (m)", min=1.40, max=2.10, value=1.70, step=0.01, help="Height in metres, e.g. 1.75."),
    "Weight": dict(label="Weight (kg)", min=30.0, max=200.0, value=70.0, step=0.5, help="Weight in kilograms."),
    "FCVC": dict(
        label="Vegetable consumption (FCVC)",
        min=1,
        max=3,
        value=2,
        step=1,
        help="Frequency of eating vegetables: 1 = never, 2 = sometimes, 3 = always.",
    ),
    "NCP": dict(
        label="Main meals per day (NCP)", min=1, max=4, value=3, step=1, help="Number of main meals eaten each day."
    ),
    "CH2O": dict(
        label="Daily water intake (CH2O)",
        min=1,
        max=3,
        value=2,
        step=1,
        help="Daily water: (1 = <1L), (2 = 1 - 2L), (3 = >2L).",
    ),
    "FAF": dict(
        label="Physical activity (FAF)",
        min=0,
        max=3,
        value=1,
        step=1,
        help="Days of physical activity per week: 0 = none … 3 = 4 - 5 days.",
    ),
    "TUE": dict(
        label="Tech device time (TUE)",
        min=0,
        max=2,
        value=1,
        step=1,
        help="Time using technology devices: 0 = 0 - 2h, 1 = 3 - 5h, 2 = >5h.",
    ),
}

CATEGORICAL_FEATURES = {
    "Gender": dict(label="Gender", options=["Female", "Male"], help="Biological sex as recorded in the dataset."),
    "family_history_with_overweight": dict(
        label="Family history of overweight",
        options=YES_NO,
        help="Has a family member suffered or suffers from overweight?",
    ),
    "FAVC": dict(
        label="High-calorie food often (FAVC)", options=YES_NO, help="Do you frequently eat high-calorie food?"
    ),
    "CAEC": dict(label="Snacking between meals (CAEC)", options=FREQ, help="How often do you eat food between meals?"),
    "SMOKE": dict(label="Smoker", options=YES_NO, help="Do you smoke?"),
    "SCC": dict(
        label="Monitors calorie intake (SCC)", options=YES_NO, help="Do you monitor the calories you eat daily?"
    ),
    "CALC": dict(label="Alcohol consumption (CALC)", options=FREQ, help="How often do you drink alcohol?"),
    "MTRANS": dict(
        label="Transport used (MTRANS)",
        options=["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
        help="Means of transport usually used.",
    ),
}

# ----------------------------------------------------------------------
# Output class metadata (readable label, description, color, sort order)
# ----------------------------------------------------------------------
CLASS_INFO = {
    "Insufficient_Weight": dict(
        label="Insufficient Weight", color="#3b82f6", desc="Body weight is below the healthy range.", order=0
    ),
    "Normal_Weight": dict(
        label="Normal Weight", color="#22c55e", desc="Body weight is within the healthy range.", order=1
    ),
    "Overweight_Level_I": dict(
        label="Overweight (Level I)", color="#eab308", desc="Mildly above the healthy weight range.", order=2
    ),
    "Overweight_Level_II": dict(
        label="Overweight (Level II)", color="#f59e0b", desc="Moderately above the healthy weight range.", order=3
    ),
    "Obesity_Type_I": dict(label="Obesity Type I", color="#f97316", desc="Class I obesity (BMI ~30-35).", order=4),
    "Obesity_Type_II": dict(label="Obesity Type II", color="#ef4444", desc="Class II obesity (BMI ~35-40).", order=5),
    "Obesity_Type_III": dict(
        label="Obesity Type III", color="#b91c1c", desc="Class III / severe obesity (BMI 40+).", order=6
    ),
}


# Class label function
def class_label(raw: str) -> str:
    return CLASS_INFO.get(raw, {}).get("label", raw.replace("_", " "))


# Class color function
def class_color(raw: str) -> str:
    return CLASS_INFO.get(raw, {}).get("color", "#64748b")


def obesity_form() -> dict:
    """Render all inputs and return a dict keyed by the raw model column names."""
    inputs = {}

    # Personal Details Section
    st.markdown("#### Personal details")
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["Gender"] = st.selectbox(
            CATEGORICAL_FEATURES["Gender"]["label"],
            CATEGORICAL_FEATURES["Gender"]["options"],
            help=CATEGORICAL_FEATURES["Gender"]["help"],
        )
    with c2:
        m = NUMERIC_FEATURES["Age"]
        inputs["Age"] = st.number_input(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
    with c3:
        inputs["family_history_with_overweight"] = st.selectbox(
            CATEGORICAL_FEATURES["family_history_with_overweight"]["label"],
            CATEGORICAL_FEATURES["family_history_with_overweight"]["options"],
            help=CATEGORICAL_FEATURES["family_history_with_overweight"]["help"],
        )

    c1, c2 = st.columns(2)
    with c1:
        m = NUMERIC_FEATURES["Height"]
        inputs["Height"] = st.number_input(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
    with c2:
        m = NUMERIC_FEATURES["Weight"]
        inputs["Weight"] = st.number_input(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])

    # Live BMI hint
    bmi = inputs["Weight"] / (inputs["Height"] ** 2)
    st.caption(f"Computed BMI: **{bmi:.1f}** kg/m²")

    # Eating Habits Section
    st.markdown("#### Eating habits")
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["FAVC"] = st.selectbox(
            CATEGORICAL_FEATURES["FAVC"]["label"],
            CATEGORICAL_FEATURES["FAVC"]["options"],
            help=CATEGORICAL_FEATURES["FAVC"]["help"],
        )
        m = NUMERIC_FEATURES["FCVC"]
        inputs["FCVC"] = st.slider(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
    with c2:
        inputs["CAEC"] = st.selectbox(
            CATEGORICAL_FEATURES["CAEC"]["label"],
            CATEGORICAL_FEATURES["CAEC"]["options"],
            help=CATEGORICAL_FEATURES["CAEC"]["help"],
        )
        m = NUMERIC_FEATURES["NCP"]
        inputs["NCP"] = st.slider(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
    with c3:
        inputs["CALC"] = st.selectbox(
            CATEGORICAL_FEATURES["CALC"]["label"],
            CATEGORICAL_FEATURES["CALC"]["options"],
            help=CATEGORICAL_FEATURES["CALC"]["help"],
        )
        inputs["SCC"] = st.selectbox(
            CATEGORICAL_FEATURES["SCC"]["label"],
            CATEGORICAL_FEATURES["SCC"]["options"],
            help=CATEGORICAL_FEATURES["SCC"]["help"],
        )

    # Lifestyle Section
    st.markdown("#### Lifestyle")
    c1, c2, c3 = st.columns(3)
    with c1:
        m = NUMERIC_FEATURES["CH2O"]
        inputs["CH2O"] = st.slider(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
        inputs["SMOKE"] = st.selectbox(
            CATEGORICAL_FEATURES["SMOKE"]["label"],
            CATEGORICAL_FEATURES["SMOKE"]["options"],
            help=CATEGORICAL_FEATURES["SMOKE"]["help"],
        )
    with c2:
        m = NUMERIC_FEATURES["FAF"]
        inputs["FAF"] = st.slider(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
    with c3:
        m = NUMERIC_FEATURES["TUE"]
        inputs["TUE"] = st.slider(m["label"], m["min"], m["max"], m["value"], m["step"], help=m["help"])
        inputs["MTRANS"] = st.selectbox(
            CATEGORICAL_FEATURES["MTRANS"]["label"],
            CATEGORICAL_FEATURES["MTRANS"]["options"],
            help=CATEGORICAL_FEATURES["MTRANS"]["help"],
        )

    return inputs
