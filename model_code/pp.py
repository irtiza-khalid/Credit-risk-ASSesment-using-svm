import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
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

# Train the SVM model
svm = SVC()
svm.fit(X_scaled, y)

# Save the trained model
pickle.dump(svm, open("svm_model.pkl", "wb"))
