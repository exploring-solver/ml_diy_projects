from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load models and scaler
logreg_model = joblib.load('models/logistic_regression_churn_model.pkl')
rf_model = joblib.load('models/random_forest_churn_model.pkl')
svm_model = joblib.load('models/svm_churn_model.pkl')
knn_model = joblib.load('models/knn_churn_model.pkl')
gb_model = joblib.load('models/gradient_boosting_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
# Load feature columns from training
feature_columns = joblib.load('models/feature_columns.pkl')

# Dictionary to map the model name from the form
model_mapping = {
    'logreg': logreg_model,
    'rf': rf_model,
    'svm': svm_model,
    'knn': knn_model,
    'gb': gb_model
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data as a JSON object
    data = request.get_json()
    
    # Extract the features from the form submission
    features = data['features']
    model_choice = data['model']
    
    # Convert the feature dictionary to a pandas DataFrame
    input_features = pd.DataFrame([{
        'tenure': features['tenure'],
        'MonthlyCharges': features['MonthlyCharges'],
        'TotalCharges': features['TotalCharges'],
        'gender': features['gender'],
        'PhoneService': features['PhoneService'],
        'PaperlessBilling': features['PaperlessBilling'],
        'FamilySize': features['FamilySize'],
        'HasMultipleServices': features['HasMultipleServices']
    }])

    # Make sure the input features have the same structure as the training set
    # Add missing columns with a default value of 0
    for col in feature_columns:
        if col not in input_features.columns:
            input_features[col] = 0

    # Ensure columns are in the same order as the training set
    input_features = input_features[feature_columns]

    # Scale the numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_features[numerical_cols] = scaler.transform(input_features[numerical_cols])
    
    # Select the model based on user input
    selected_model = model_mapping[model_choice]
    
    # Make predictions
    churn_prediction = selected_model.predict(input_features)[0]
    churn_probability = selected_model.predict_proba(input_features)[0][1]  # Probability of churn
    
    # Send back the prediction results
    result = {
        'churn_prediction': int(churn_prediction),  # 0 or 1
        'churn_probability': float(churn_probability)  # Probability of churn
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
