import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Breast Cancer Prediction App")
st.write("Enter values for each of the 30 features used by the model.")

# List of feature names (in order)
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Get user input for each feature
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature.title()}", format="%.5f")
    input_data.append(val)

# Predict when button is clicked
if st.button("Predict"):
    input_array = np.array([input_data])  # Shape (1, 30)
    prediction = model.predict(input_array)
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.success(f"The predicted result is: **{result}**")
