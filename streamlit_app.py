import streamlit as st
import pickle
import numpy as np

# Load the trained model and symptom list
with open("medlens_pro_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("medlens_pro_symptom_classes.pkl", "rb") as f:
    symptoms = pickle.load(f)

st.title("🧠 MedLens Pro: AI Health Assistant")

input_text = st.text_input("Enter symptoms (comma-separated):")

if st.button("Predict"):
    input_symptoms = [sym.strip().lower() for sym in input_text.split(",")]

    # Create input vector based on known symptoms
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptoms]

    try:
        prediction = model.predict([input_vector])[0]
        st.success(f"🩺 Predicted Disease: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
