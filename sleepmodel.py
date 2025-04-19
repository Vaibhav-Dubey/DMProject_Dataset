import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
file_path1 = "C:\\Users\\saira\\Downloads\\mturkfitbit_export_3.12.16-4.11.16\\Fitabase Data 3.12.16-4.11.16\\processed_sleep_data3-4.csv"
file_path2 = "C:\\Users\\saira\\Downloads\\mturkfitbit_export_4.12.16-5.12.16\\Fitabase Data 4.12.16-5.12.16\\processed_sleep_data_4-5.csv"

# Read the datasets and merge
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
sleep_data = pd.concat([data1, data2])

# Convert date column to datetime
sleep_data['date'] = pd.to_datetime(sleep_data['date'])

# Compute monthly average sleep minutes per user
monthly_avg_sleep = sleep_data.groupby('Id')['total_sleep_minutes'].mean().reset_index()

# Define sleep thresholds
optimal_min, optimal_max = 420, 540  # Ideal sleep range in minutes
insufficient_sleep, oversleep_threshold = 360, 600

# Feature Engineering
monthly_avg_sleep['deviation'] = monthly_avg_sleep['total_sleep_minutes'].apply(
    lambda x: min(abs(x - optimal_min), abs(x - optimal_max))
)

# Apply a weighted function to deviation score
weight_exponent = 1.5  # Adjust this exponent to control sensitivity
monthly_avg_sleep['weighted_deviation'] = (monthly_avg_sleep['deviation'] ** weight_exponent)

# Normalize weighted deviation score
max_weighted_dev = monthly_avg_sleep['weighted_deviation'].max()
monthly_avg_sleep['deviation_score'] = 100 - (monthly_avg_sleep['weighted_deviation'] / max_weighted_dev) * 100
monthly_avg_sleep['deviation_score'] = monthly_avg_sleep['deviation_score'].clip(0, 100)

# Sleep Score (Now based on weighted deviation score)
monthly_avg_sleep['sleep_score'] = monthly_avg_sleep['deviation_score']

# Prepare Features and Target
features = ['total_sleep_minutes', 'deviation_score']
target = 'sleep_score'
X = monthly_avg_sleep[features].dropna()
y = monthly_avg_sleep[target].dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_pipeline, "monthly_sleep_score_model.pkl")

print("Model trained and saved as monthly_sleep_score_model.pkl")
