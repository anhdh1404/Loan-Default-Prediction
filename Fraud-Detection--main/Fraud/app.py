


from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('C:\\Users\\dell\\Desktop\\Project\\Fraud\\best_model.pkl')
scaler = joblib.load('C:\\Users\\dell\\Desktop\\Project\\Fraud\\scaler.pkl')

# Get the feature names from the model (if available)
try:
    # For scikit-learn models
    expected_columns = model.feature_names_in_
except AttributeError:
    # Fallback to your original list if feature_names_in_ not available
    expected_columns = [
        'LOAN', 'MORTDUE', 'VALUE', 'DEROG', 'DELINQ', 
        'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'YOJ',
        'JOB_Office', 'JOB_Other', 'JOB_ProfExe', 'JOB_Sales', 'JOB_Self'
    ]

# Job options for the form
job_options = [
    'Office', 'Other', 'ProfExe', 'Sales', 'Self'
]

@app.route('/')
def home():
    return render_template('index.html', job_options=job_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Create a DataFrame with all expected columns initialized to 0
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        # Fill in the numerical values
        numerical_cols = ['LOAN', 'MORTDUE', 'VALUE', 'DEROG', 'DELINQ', 
                        'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'YOJ']
        for col in numerical_cols:
            if col in input_data.columns:
                input_data[col] = float(form_data.get(col, 0))
        
        # Handle JOB one-hot encoding
        selected_job = form_data.get('JOB', 'Other')
        job_col = f"JOB_{selected_job}"
        if job_col in input_data.columns:
            input_data[job_col] = 1
        
        # Ensure columns are in correct order
        input_data = input_data[expected_columns]
        
        # Scale the data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Prepare result
        result = {
            'prediction': 'HIGH RISK (BAD)' if prediction == 1 else 'LOW RISK (GOOD)',
            'probability': f"{probability*100:.2f}%",
            'input_data': form_data
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('index.html', 
                             error=str(e),
                             job_options=job_options)

if __name__ == '__main__':
    app.run(debug=True)