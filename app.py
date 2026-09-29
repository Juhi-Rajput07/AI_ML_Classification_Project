import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wine Class Prediction",
    page_icon="🍷",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = Path("models/random_forest_model.joblib")
FEATURE_PATH = Path("models/feature_names.joblib")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# Dataset-based default values
# --------------------------------------------------

default_values = {
    "alcohol": 13.0,
    "malic_acid": 2.3,
    "ash": 2.3,
    "alcalinity_of_ash": 20.0,
    "magnesium": 100.0,
    "total_phenols": 2.3,
    "flavanoids": 2.0,
    "nonflavanoid_phenols": 0.35,
    "proanthocyanins": 1.5,
    "color_intensity": 5.0,
    "hue": 1.0,
    "od280/od315_of_diluted_wines": 2.5,
    "proline": 750.0
}


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🍷 Wine Class Prediction System")

st.write(
    "An AI-based multiclass classification application "
    "using a Random Forest machine learning model."
)

st.info(
    "Enter the 13 wine-related features below and click "
    "'Predict Wine Class' to generate a prediction."
)


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.header("Enter Wine Features")

inputs = {}

columns = st.columns(3)

for index, feature in enumerate(feature_names):
    with columns[index % 3]:
        inputs[feature] = st.number_input(
            feature,
            value=float(default_values.get(feature, 0.0)),
            format="%.4f"
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Wine Class", type="primary"):

    input_data = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    predicted_class = f"class_{prediction}"

    st.success(
        f"Predicted Wine Class: **{predicted_class}**"
    )

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Class": [
            f"class_{i}"
            for i in range(len(probabilities))
        ],
        "Probability": probabilities
    })

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.dataframe(
        probability_df,
        use_container_width=True
    )

    st.caption(
        "Prediction generated using the trained Random Forest classifier."
    )


# --------------------------------------------------
# Project information
# --------------------------------------------------

st.divider()

st.subheader("About the Project")

st.write(
    """
    This application demonstrates an end-to-end machine learning workflow.
    The Wine dataset is analysed, cleaned, explored, preprocessed and used
    for multiclass classification.

    Two modelling approaches were studied during the project:
    Random Forest and a basic Deep Neural Network.

    The deployed application uses the Random Forest classifier.
    """
)