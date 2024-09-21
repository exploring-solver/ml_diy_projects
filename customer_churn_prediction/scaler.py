import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('data/customer_churn_data.csv')

# Drop irrelevant columns and preprocess the data similarly to your previous steps
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encoding and one-hot encoding (as you did before)
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0})

categorical_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature Scaling for numerical features
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Save the scaler to a pkl file
joblib.dump(scaler, 'backend/models/scaler.pkl')
print("Scaler has been saved as 'scaler.pkl'")
