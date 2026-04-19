import streamlit as st
import numpy as np
import joblib
import gdown
import os

st.title("Student Mental Health System")
st.title("Burnout Prediction System")

# --------------------------
# LOAD BURNOUT MODEL
# --------------------------
BURNOUT_PATH = "burnout_model.pkl"

if not os.path.exists(BURNOUT_PATH):
    url = "https://drive.google.com/uc?export=download&id=1xRJmEbQw6zahs7XWPjeugVud3WnK_7bt"
    gdown.download(url, BURNOUT_PATH, quiet=False)

burnout_model = joblib.load(BURNOUT_PATH)

# --------------------------
# LOAD STRESS MODEL (must exist in repo OR same method)
# --------------------------
STRESS_PATH = "stress_model.pkl"

if os.path.exists(STRESS_PATH):
    stress_model = joblib.load(STRESS_PATH)
else:
    stress_model = None

# --------------------------
# UI
# --------------------------
st.header("Enter Student Details")

age = st.number_input("Age")
sleep = st.number_input("Sleep Hours")
study = st.number_input("Study Hours")
stress_input = st.number_input("Stress Score")
anxiety = st.number_input("Anxiety Score")

if st.button("Predict"):

    b_input = np.zeros(burnout_model.n_features_in_)
    b_input[:5] = [age, sleep, study, stress_input, anxiety]

    burnout_pred = burnout_model.predict([b_input])[0]

    st.subheader("Results:")
    st.write("Burnout Level:", burnout_pred)

    if stress_model:
        s_input = np.zeros(stress_model.n_features_in_)
        s_input[:5] = [age, sleep, study, stress_input, anxiety]

        stress_pred = stress_model.predict([s_input])[0]
        st.write("Stress Level:", stress_pred)