"""
app.py
------
Interactive Streamlit dashboard. A user enters their own habits via
sliders, gets an instant stress-level prediction, AND sees a SHAP
explanation of why the model predicted that for them specifically.

Run:
    streamlit run app.py
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Student Stress Predictor", page_icon="🧠")

# Load model artifacts
model = joblib.load("model/rf_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_names = joblib.load("model/feature_names.pkl")
explainer = shap.TreeExplainer(model)

st.title("🧠 Student Stress Level Predictor")
st.write(
    "Enter your habits below. The model predicts your stress level "
    "**and explains which factors influenced the prediction the most.**"
)

# --- Input widgets ---
col1, col2 = st.columns(2)
with col1:
    sleep_hours = st.slider("Sleep hours/night", 2.0, 10.0, 6.5, 0.5)
    study_hours = st.slider("Study hours/day", 0.0, 12.0, 4.0, 0.5)
    screen_time = st.slider("Screen time/day (hrs)", 0.0, 14.0, 6.0, 0.5)
with col2:
    social_activity = st.slider("Social activity (0-10)", 0, 10, 5)
    academic_pressure = st.slider("Academic pressure (1-5)", 1, 5, 3)
    extracurricular_hours = st.slider("Extracurricular hrs/day", 0.0, 8.0, 2.0, 0.5)

input_df = pd.DataFrame([{
    "sleep_hours": sleep_hours,
    "study_hours": study_hours,
    "screen_time": screen_time,
    "social_activity": social_activity,
    "academic_pressure": academic_pressure,
    "extracurricular_hours": extracurricular_hours,
}])[feature_names]

# --- Predict button ---
if st.button("Predict my stress level"):
    prediction = model.predict(input_df)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.subheader(f"Predicted stress level: **{predicted_label}**")

    # SHAP explanation for this single prediction
    shap_values = explainer.shap_values(input_df)
    class_index = list(label_encoder.classes_).index(predicted_label)

    st.write("### Why this prediction?")
    fig = plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0, :, class_index],
            base_values=explainer.expected_value[class_index],
            data=input_df.iloc[0],
            feature_names=feature_names,
        ),
        show=False,
    )
    st.pyplot(fig)

    st.caption(
        "Bars pushing right increase your stress-risk score; "
        "bars pushing left decrease it."
    )
