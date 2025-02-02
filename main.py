import uvicorn
from fastapi import FastAPI
from info import info_data
import pickle 
import numpy as np

app = FastAPI()

# Load the trained model
with open('model.pkl', 'rb') as pickle_in:
    model = pickle.load(pickle_in)

# Load the fitted scaler
with open('scaler.pkl', 'rb') as pickle_in:
    scaler = pickle.load(pickle_in)

@app.post('/predict')
def predict(data: info_data):
    age = data.age
    income = data.income
    emp_len = data.employment_len
    amnt = data.loan_amnt
    int_rate = data.loan_int_rate
    percent_inc = data.loan_percent_income
    cred_len = data.cred_hist_len

    ownership_categories = ['OTHER', 'OWN', 'RENT']
    loan_intent_categories = ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    
    # One-hot encode ownership
    ownership_encoding = [1 if category == data.ownership else 0 for category in ownership_categories]
    
    # One-hot encode loan intent
    loan_intent_encoding = [1 if category == data.loan_intent else 0 for category in loan_intent_categories]

    # Prepare data to scale (reshaping to 2D)
    data_to_scale = np.array([[age, income, emp_len, amnt, int_rate, percent_inc, cred_len]])
    
    # Scale the input features using the loaded scaler
    scaled_data = scaler.transform(data_to_scale)

    # Prepare the input for prediction
    prediction_input = [
        *scaled_data[0],  # Unpack the scaled features
        *ownership_encoding,
        *loan_intent_encoding
    ]

    # Make prediction using the model
    prediction = model.predict_proba([prediction_input])
    
    return {'prob of eligible': str(prediction[0][0]), 'prob of not eligible': str(prediction[0][1])}

if __name__ == '__main__':
    uvicorn.run(app)
