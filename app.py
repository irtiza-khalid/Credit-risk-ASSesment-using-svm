import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import pickle

app = Flask(__name__)

# Load the trained SVM model
svm = pickle.load(open(r"svm_model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the data
    data = pd.read_csv(r"C:\Users\omer0\OneDrive\Desktop\credit_risk_dataset.csv")

    # Drop unnecessary columns
    data.drop(['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis=1, inplace=True)

    # Convert categorical labels to numerical values
    label_encoder = LabelEncoder()
    data['loan_status'] = label_encoder.fit_transform(data['loan_status'])

    # Separate features and target variable
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Get user input from the form
    user_input = [
        float(request.form['person_age']),
        float(request.form['person_income']),
        float(request.form['person_emp_length']),
        float(request.form['loan_amnt']),
        float(request.form['loan_int_rate']),
        float(request.form['loan_percent_income']),
        float(request.form['cb_person_cred_hist_length'])
    ]

    # Make a prediction on the user input
    user_input_scaled = scaler.transform([user_input])
    prediction = svm.predict(user_input_scaled)[0]

    # Add the user input to the dataset for calculating the percentage of risky loans
    X_user = np.concatenate((X_scaled, user_input_scaled))
    y_user = np.append(y, prediction)

    # Calculate percentage of risky loans
    percentage_risk = (sum(y_user) / len(y_user)) * 100

    return render_template('index.html', prediction_result=prediction, percentage_risk=percentage_risk)

if __name__ == '__main__':
    app.run(debug=True)
