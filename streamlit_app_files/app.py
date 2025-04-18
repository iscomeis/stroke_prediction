import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load your trained model and required files
with open("stroke_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("dummies.pkl", "rb") as dummies_file:
    columns = pickle.load(dummies_file)

# Custom styles for user-friendly design
page_bg = """
<style>
body {
    background-color: #f9f9f9; /* Light background for readability */
    font-family: 'Arial', sans-serif;
}
h1, h2, h3 {
    color: #333333;
}
button {
    background-color: #007BFF;
    color: white;
    border-radius: 5px;
    padding: 10px 15px;
    font-size: 16px;
    cursor: pointer;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App Title and Description
st.title("🩺 Stroke Prediction App")
st.markdown("""
Welcome to the **Stroke Prediction App**!  
Provide your details below, and our machine learning model will predict the likelihood of a stroke.  
""")

# Input fields for user data
st.markdown("### Fill out your details:")

# Residence Type
residence = st.radio("Where do you live?", ("Urban", "Rural"))

# Smoking Status
smoking_status = st.selectbox("What is your smoking status?", ("never smoked", "formerly smoked", "smokes"))

# Age
age = st.number_input("Enter your age:", min_value=18, max_value=82, step=1)

# Gender
gender = st.selectbox("What is your gender?", ("Male", "Female", "Other"))

# Ever Married
ever_married = st.radio("Have you ever been married?", ("Yes", "No"))

# Hypertension
hypertension = st.radio("Do you have hypertension?", ("Yes", "No"))

# Heart Disease
heart_disease = st.radio("Do you have heart disease?", ("Yes", "No"))

# Avg Glucose Level
avg_glucose_level = st.number_input("Enter your average glucose level:", min_value=55.0, max_value=267.0, step=0.1)

# Work Type
work_type = st.selectbox("What is your work type?", 
                         ("Private", "Self-employed", "Govt_job", "children", "Never_worked"))

# Prediction Logic
if st.button("🧮 Predict Stroke"):
    # Ensure all inputs are provided
    if age and avg_glucose_level:
        # Preprocess the inputs to align with training data structure
        input_df = pd.DataFrame([{
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "avg_glucose_level": avg_glucose_level,
            "gender_Male": 1 if gender == "Male" else 0,
            "gender_Other": 1 if gender == "Other" else 0,
            "ever_married_Yes": 1 if ever_married == "Yes" else 0,
            "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
            "work_type_Private": 1 if work_type == "Private" else 0,
            "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
            "work_type_children": 1 if work_type == "children" else 0,
            "Residence_type_Urban": 1 if residence == "Urban" else 0,
            "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
            "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
            "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
        }])

        # Align features with the training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Convert to numpy array for prediction
        input_data = input_df.values

        # Perform the prediction
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.error("⚠️ **High risk of stroke** detected. Please consult your doctor.")
        else:
            st.success("✅ **Low risk of stroke** detected. Stay healthy!")
    else:
        st.error("⚠️ Please ensure all fields are filled!")

# Footer Disclaimer
st.markdown("---")
st.markdown("""
<sub>
⚠️ **Disclaimer:** This application is intended for demonstration purposes only.  
It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult your physician or other qualified health provider with any questions regarding a medical condition.  
</sub>
""", unsafe_allow_html=True)
