import streamlit as st
import numpy as np
import joblib
import os

st.title("Student Mental Health & Burnout Predictor")

# --------------------------
# INPUTS
# --------------------------
st.header("Enter Student Details")

age = st.number_input("Age")
sleep = st.number_input("Sleep Hours")
study = st.number_input("Study Hours")
stress_input = st.number_input("Stress Score")
anxiety = st.number_input("Anxiety Score")

# PREDICT BUTTON
if st.button("Predict"):

    try:
        #  LOAD MODELS ONLY WHEN NEEDED)
        burnout_model = joblib.load("burnout_model.pkl")

        # Optional stress model
        if os.path.exists("stress_model.pkl"):
            stress_model = joblib.load("stress_model.pkl")
        else:
            stress_model = None

        # INPUT FORMAT 
        b_input = np.array([[age, sleep, study, stress_input, anxiety]])

        burnout_pred = burnout_model.predict(b_input)[0]

        st.subheader("Results")
        st.success(f"Burnout Level: {burnout_pred}")

        # STRESS PREDICTION (OPTIONAL) 
        if stress_model:
            s_input = np.array([[age, sleep, study, stress_input, anxiety]])
            stress_pred = stress_model.predict(s_input)[0]

            st.info(f"Stress Level: {stress_pred}")
        else:
            st.warning("Stress model not found")

    except Exception as e:
        st.error(f"Error: {e}")