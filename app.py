from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

model_path = "best_loan_pred_model.pkl"
model = joblib.load(model_path)

st.title("Customers Loan Predicition")
st.write("Fill in your information correctly to get automated insights for your loan approval")

# Create user field
Gender = st.selectbox("What is your gender?", ["Male", "Female"])
Married = st.selectbox("Are you married?", ["Yes", "No"])                         
Dependents =st.selectbox("How many dependents do you have?", ["0", "1", "2", "3+"])
Graduate = st.selectbox("Are you a university graduate?", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Are you self_Employed?", ["Yes", "No"])
ApplicantIncome = st.number_input("How much do you earn?", min_value=0)
CoapplicantIncome = st.number_input("How much is your coapplicant income?", min_value=0)
LoanAmount = st.number_input("What is your required loan amount?", min_value=0)
Loan_Amount_Term = st.selectbox("How long do you want the loan for (in day)?", [480, 360, 180, 120, 60])
Credit_History = st.selectbox("What is your credit history?", [1, 0])
Property_Area = st.selectbox("What property area do you reside in?", ["Urban", "Rural", "Semiurban"])

Total_Income = ApplicantIncome + CoapplicantIncome

# Manual Label Encoding

categorical_mappings = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Graduate": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
}

#Apply the Encoded data

encoded_data = [
    categorical_mappings["Gender"][Gender],
    categorical_mappings["Married"][Married],
    categorical_mappings["Dependents"][Dependents],
    categorical_mappings["Graduate"][Graduate],
    categorical_mappings["Self_Employed"][Self_Employed],
    ApplicantIncome, 
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    categorical_mappings["Property_Area"][Property_Area],
    Total_Income
]

#Define the input_data
input_data = np.array([encoded_data])


#Standardize the numerical_values in the input_data
scaler = StandardScaler()
numerical_indices = [5, 6, 7, 8, 9, 11]
input_data[:, numerical_indices] = scaler.fit_transform(input_data[:, numerical_indices])


#Define the model prediction and include button
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.success(f"Loan Status: {result}")










#trial_input_2 = np.array([[0, #Gender (1 = Male, 0 = Female)
                          #1, #Married (1 = Yes, 0 = No)
                          #3, #Dependents (e.g., 0, 1, 2, 3)
                          #1, #Graduate (1 = Graduate, 0 = Not Graduate)
                          #1, #Self_Employed (1 = Yes, 0 = No)
                          #ApplicantIncome, 
                          #CoapplicantIncome,
                          #5000, #LoanAmount
                          #360, #Loan_Amount_Term
                          #1, # Credit_History (1 = Good, 0 = Bad)
                          #2, #Property_Area (0 = Urban, 1 = Rural, 2 = Semiurban)
                          #Total_Income
#]])


