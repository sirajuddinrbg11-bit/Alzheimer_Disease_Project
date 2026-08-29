
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and the scaler
# Ensure these paths are correct if deploying to a different environment
try:
    model = joblib.load('best_logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl') # Load the fitted scaler
except FileNotFoundError as e:
    st.error(f"Error loading file: {e}. Please ensure both 'best_logistic_regression_model.pkl' and 'scaler.pkl' are in the same directory as app.py.")
    st.stop()

# Title of the app
st.title('Alzheimer's Disease Diagnosis Predictor')
st.write('Enter patient details to predict the likelihood of Alzheimer's Disease.')

# Sidebar for user inputs
st.sidebar.header('Patient Input Features')

def user_input_features():
    # These are derived from `X.columns` from the notebook
    age = st.sidebar.slider('Age', 60, 90, 70)
    gender = st.sidebar.selectbox('Gender (0=Female, 1=Male)', [0, 1])
    ethnicity = st.sidebar.selectbox('Ethnicity (0-3)', [0, 1, 2, 3]) # Assuming 0-3 are encoded categories
    education_level = st.sidebar.selectbox('Education Level (0-3)', [0, 1, 2, 3])
    bmi = st.sidebar.slider('BMI', 15.0, 40.0, 25.0)
    smoking = st.sidebar.selectbox('Smoking (0=No, 1=Yes)', [0, 1])
    alcohol_consumption = st.sidebar.slider('Alcohol Consumption', 0.0, 20.0, 10.0)
    physical_activity = st.sidebar.slider('Physical Activity (hours/week)', 0.0, 10.0, 5.0)
    diet_quality = st.sidebar.slider('Diet Quality', 0.0, 10.0, 5.0)
    sleep_quality = st.sidebar.slider('Sleep Quality', 0.0, 10.0, 5.0)
    family_history = st.sidebar.selectbox('Family History of Alzheimer's (0=No, 1=Yes)', [0, 1])
    cardiovascular_disease = st.sidebar.selectbox('Cardiovascular Disease (0=No, 1=Yes)', [0, 1])
    diabetes = st.sidebar.selectbox('Diabetes (0=No, 1=Yes)', [0, 1])
    depression = st.sidebar.selectbox('Depression (0=No, 1=Yes)', [0, 1])
    head_injury = st.sidebar.selectbox('Head Injury (0=No, 1=Yes)', [0, 1])
    hypertension = st.sidebar.selectbox('Hypertension (0=No, 1=Yes)', [0, 1])
    systolic_bp = st.sidebar.slider('Systolic BP', 90, 180, 120)
    diastolic_bp = st.sidebar.slider('Diastolic BP', 60, 120, 80)
    cholesterol_total = st.sidebar.slider('Cholesterol Total', 100.0, 300.0, 200.0)
    cholesterol_ldl = st.sidebar.slider('Cholesterol LDL', 50.0, 200.0, 100.0)
    cholesterol_hdl = st.sidebar.slider('Cholesterol HDL', 20.0, 100.0, 50.0)
    cholesterol_triglycerides = st.sidebar.slider('Cholesterol Triglycerides', 50.0, 400.0, 150.0)
    mmse = st.sidebar.slider('MMSE Score', 0.0, 30.0, 25.0)
    functional_assessment = st.sidebar.slider('Functional Assessment', 0.0, 10.0, 5.0)
    memory_complaints = st.sidebar.selectbox('Memory Complaints (0=No, 1=Yes)', [0, 1])
    behavioral_problems = st.sidebar.selectbox('Behavioral Problems (0=No, 1=Yes)', [0, 1])
    adl = st.sidebar.slider('ADL Score', 0.0, 10.0, 5.0)
    confusion = st.sidebar.selectbox('Confusion (0=No, 1=Yes)', [0, 1])
    disorientation = st.sidebar.selectbox('Disorientation (0=No, 1=Yes)', [0, 1])
    personality_changes = st.sidebar.selectbox('Personality Changes (0=No, 1=Yes)', [0, 1])
    difficulty_completing_tasks = st.sidebar.selectbox('Difficulty Completing Tasks (0=No, 1=Yes)', [0, 1])
    forgetfulness = st.sidebar.selectbox('Forgetfulness (0=No, 1=Yes)', [0, 1])

    data = {
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'EducationLevel': education_level,
        'BMI': bmi,
        'Smoking': smoking,
        'AlcoholConsumption': alcohol_consumption,
        'PhysicalActivity': physical_activity,
        'DietQuality': diet_quality,
        'SleepQuality': sleep_quality,
        'FamilyHistoryAlzheimers': family_history,
        'CardiovascularDisease': cardiovascular_disease,
        'Diabetes': diabetes,
        'Depression': depression,
        'HeadInjury': head_injury,
        'Hypertension': hypertension,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'CholesterolTotal': cholesterol_total,
        'CholesterolLDL': cholesterol_ldl,
        'CholesterolHDL': cholesterol_hdl,
        'CholesterolTriglycerides': cholesterol_triglycerides,
        'MMSE': mmse,
        'FunctionalAssessment': functional_assessment,
        'MemoryComplaints': memory_complaints,
        'BehavioralProblems': behavioral_problems,
        'ADL': adl,
        'Confusion': confusion,
        'Disorientation': disorientation,
        'PersonalityChanges': personality_changes,
        'DifficultyCompletingTasks': difficulty_completing_tasks,
        'Forgetfulness': forgetfulness
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Apply scaling to the input features
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

st.subheader('Prediction')
alzheimers_diagnosis = np.array(['No Alzheimer's', 'Alzheimer's']) # Using np.array for indexing
st.write(alzheimers_diagnosis[prediction][0])

st.subheader('Prediction Probability')
st.write(f"No Alzheimer's: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Alzheimer's: {prediction_proba[0][1]*100:.2f}%")
