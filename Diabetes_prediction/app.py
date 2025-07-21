import streamlit as st
import numpy as np
import joblib

# ------------------ Load Model and Scaler ------------------
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ Feature Names ------------------
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure',
                 'SkinThickness', 'Insulin', 'BMI', 'Age']

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🧬")
st.title("🩺 Diabetes Prediction using Random Forest")
st.write("This app uses a trained Random Forest model to predict the likelihood of diabetes.")

st.markdown("---")
st.subheader("📋 Enter Patient Information")

# ------------------ Input Form ------------------
with st.form("input_form"):
    user_input = []
    for feature in feature_names:
        value = st.number_input(f"{feature}:", min_value=0.0, step=0.1)
        user_input.append(value)
    submitted = st.form_submit_button("🔍 Predict")

# ------------------ Make Prediction ------------------
if submitted:
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ The model predicts that the person **has diabetes**.\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ The model predicts that the person **does not have diabetes**.\n\nProbability: {1 - probability:.2f}")
