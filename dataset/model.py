# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the raw dataset
file_path = 'processed_ds1_grouped_activity_per_user.csv'
raw_data = pd.read_csv(file_path)

# Drop unnecessary columns (id and date)
raw_data = raw_data.drop(['id', 'date'], axis=1)

# Drop rows with any missing values
raw_data = raw_data.dropna()

# Prepare features and target
# Use raw features directly and set 'Calories' as the target
X_raw = raw_data.drop(['Calories'], axis=1).values
y_raw = (raw_data['Calories'] > raw_data['Calories'].median()).astype(int)

# Train/Test Split (80-20)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Model Evaluation Function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Logistic Regression (Revised Approach)
logreg_raw = LogisticRegression(random_state=42, max_iter=1000)
logreg_raw_metrics = evaluate_model(logreg_raw, X_train_raw, y_train_raw, X_test_raw, y_test_raw)

# Random Forest (Revised Approach)
rf_raw = RandomForestClassifier(random_state=42, n_estimators=100)
rf_raw_metrics = evaluate_model(rf_raw, X_train_raw, y_train_raw, X_test_raw, y_test_raw)

# Support Vector Machine (Revised Approach)
svm_raw = SVC(random_state=42, kernel='rbf')
svm_raw_metrics = evaluate_model(svm_raw, X_train_raw, y_train_raw, X_test_raw, y_test_raw)

# Compile the results into a DataFrame
model_performance_revised = pd.DataFrame({
    "Model": ["Logistic Regression (Revised)", "Random Forest (Revised)", "SVM (Revised)"],
    "Accuracy": [logreg_raw_metrics[0], rf_raw_metrics[0], svm_raw_metrics[0]],
    "Precision": [logreg_raw_metrics[1], rf_raw_metrics[1], svm_raw_metrics[1]],
    "Recall": [logreg_raw_metrics[2], rf_raw_metrics[2], svm_raw_metrics[2]],
    "F1 Score": [logreg_raw_metrics[3], rf_raw_metrics[3], svm_raw_metrics[3]]
})

# Print the model performance results
print("Revised Model Performance using Raw Features:")
print(model_performance_revised)
