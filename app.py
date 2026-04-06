import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =========================
# Load Saved Files
# =========================
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# =========================
# Prediction Function (Pipeline)
# =========================
def predict_new_customer(data_dict):
    new_df = pd.DataFrame([data_dict])
    
    # One-hot encoding
    new_df = pd.get_dummies(new_df)
    
    # 🔥 Bool → Int conversion (IMPORTANT)
    bool_cols = new_df.select_dtypes(include='bool').columns
    new_df[bool_cols] = new_df[bool_cols].astype(int)
    
    # Align columns
    new_df = new_df.reindex(columns=columns, fill_value=0)
    
    # Scaling
    new_scaled = scaler.transform(new_df)
    
    # Prediction
    pred = model.predict(new_scaled)
    pred_prob = model.predict(new_scaled)[0][0]
    print("value is : ",pred_prob)
    
    return int(pred > 0.3)

# =========================
# UI Design
# =========================
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("💳 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# =========================
# Inputs
# =========================

credit = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", value=50000.0)
products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
salary = st.number_input("Estimated Salary", value=50000.0)

has_card = st.selectbox("Has Credit Card", [1, 0])
active = st.selectbox("Is Active Member", [1, 0])

geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# Predict Button
# =========================

if st.button("🔍 Predict"):
    
    data = {
        "CreditScore": credit,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "Geography": geo,
        "Gender": gender
    }
    
    result = predict_new_customer(data)
    
    # =========================
    # Output
    # =========================
    
    if result == 1:
        st.error("❌ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")