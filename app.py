import streamlit as st
import numpy as np
import joblib
import gdown

st.title(" Student Mental Health System")
st.title("Burnout Prediction System")
MODEL_PATH = "burnout_model.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?export=download&id=1xRJmEbQw6zahs7XWPjeugVud3WnK_7bt"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)


# Load models
burnout_model = joblib.load("burnout_model.pkl")
stress_model = joblib.load("stress_model.pkl")

st.header("Enter Student Details")

age = st.number_input("Age")
sleep = st.number_input("Sleep Hours")
study = st.number_input("Study Hours")
stress_input = st.number_input("Stress Score")
anxiety = st.number_input("Anxiety Score")

if st.button("Predict"):

    # Create input (auto match feature size)
    b_input = np.zeros(burnout_model.n_features_in_)
    s_input = np.zeros(stress_model.n_features_in_)

    # Fill common values
    for arr in [b_input, s_input]:
        arr[0] = age
        arr[1] = sleep
        arr[2] = study
        arr[3] = stress_input
        arr[4] = anxiety

    # Predictions
    burnout_pred = burnout_model.predict([b_input])[0]
    stress_pred = stress_model.predict([s_input])[0]

    st.subheader("Results:")

    st.write(" Burnout Level:", burnout_pred)
    st.write(" Stress Level:", stress_pred)