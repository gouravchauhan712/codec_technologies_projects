import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
model = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
scaler = os.path.join(BASE_DIR, "scaler.pkl") 

# Load saved model and scaler
# model = joblib.load("logistic_regression_model.pkl")
# scaler = joblib.load("scaler.pkl")

# Get expected feature names from the model itself
feature_names = model.feature_names_in_.tolist()  # ✅ No need to load separately

st.title("📉 Customer Churn Prediction (Logistic Regression)")

# User input
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.slider("Monthly Charges", 0.0, 120.0, 60.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 1500.0)

# Manual encoding
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0
}

input_df = pd.DataFrame([input_data])

# Scale numeric features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# ✅ Reindex based on model's training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error(f"❌ The customer is likely to churn.\n\nChurn Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ The customer is not likely to churn.\n\nChurn Probability: **{probability:.2%}**")


