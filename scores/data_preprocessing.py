import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess_data():
    """
    Load and preprocess the data, returning training and testing datasets
    """
    # Create a directory for datasets if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Load the original datasets
    activity_df = pd.read_csv("../Fixed_Merged_Activity_Weight.csv")
    sleep_df = pd.read_csv("../preprocessed_sleep_data.csv")
    heart_rate_df = pd.read_csv("../Daily_Heart_Rate_All_IDs.csv")

    # Convert date columns to datetime
    activity_df['ActivityDate'] = pd.to_datetime(activity_df['ActivityDate'])
    sleep_df['date'] = pd.to_datetime(sleep_df['date'])
    heart_rate_df['Time'] = pd.to_datetime(heart_rate_df['Time'])

    # Aggregate heart rate data by day and user
    heart_rate_df['date'] = heart_rate_df['Time'].dt.date
    daily_heart_rate = heart_rate_df.groupby(['Id', 'date'])['Value'].mean().reset_index()
    daily_heart_rate['date'] = pd.to_datetime(daily_heart_rate['date'])
    daily_heart_rate = daily_heart_rate.rename(columns={'Value': 'HeartRate'})

    # Merge all datasets
    merged_df = pd.merge(activity_df, sleep_df, how='left', left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'])
    merged_df = pd.merge(merged_df, daily_heart_rate, how='left', left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'])

    # Fill missing values
    merged_df['total_sleep_minutes'] = merged_df.groupby('Id')['total_sleep_minutes'].ffill()
    merged_df['HeartRate'] = merged_df.groupby('Id')['HeartRate'].ffill()

    # Calculate established health scores
    # 1. Health Risk Score (HRS)
    weights = {
        'BMI': 0.3,
        'Sedentary': 0.2,
        'Steps': 0.2,
        'Calories': 0.1,
        'Sleep': 0.1,
        'HeartRate': 0.1
    }

    merged_df['HRS'] = (
        weights['BMI'] * merged_df['BMI'] +
        weights['Sedentary'] * (merged_df['SedentaryMinutes'] / 1000) +
        weights['Steps'] * (merged_df['TotalSteps'] / 10000) +
        weights['Calories'] * (merged_df['Calories'] / 2000) +
        weights['Sleep'] * (merged_df['total_sleep_minutes'] / 480) +
        weights['HeartRate'] * (merged_df['HeartRate'] / 60)
    )

    # 2. Framingham Risk Score
    merged_df['FraminghamRiskScore'] = (
        merged_df['BMI'] +
        (merged_df['SedentaryMinutes'] / 1000) -
        (merged_df['TotalSteps'] / 10000) +
        (merged_df['total_sleep_minutes'] / 480) -
        (merged_df['HeartRate'] / 60)
    )

    # 3. Physical Activity Index
    merged_df['PAI'] = (merged_df['VeryActiveMinutes'] * 2) + (merged_df['FairlyActiveMinutes'] * 1.5) + (merged_df['LightlyActiveMinutes'] * 1)

    # Calculate combined health score from established metrics
    merged_df['CombinedHealthScore'] = (
        0.4 * merged_df['HRS'] +
        0.3 * merged_df['FraminghamRiskScore'] +
        0.3 * merged_df['PAI']
    )

    # Print distribution of health scores
    print("\nHealth Score Distribution:")
    print(merged_df['CombinedHealthScore'].describe())

    # Plot distribution of health scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x='CombinedHealthScore', bins=30)
    plt.title('Distribution of Combined Health Scores')
    plt.show()

    def categorize_health_status(score):
        """
        Categorize health status based on the combined health score
        Using percentiles to ensure balanced categories
        """
        if score < merged_df['CombinedHealthScore'].quantile(0.2):
            return 0  # Sedentary (Unhealthy)
        elif score < merged_df['CombinedHealthScore'].quantile(0.4):
            return 1  # Low Active (Moderately Healthy)
        elif score < merged_df['CombinedHealthScore'].quantile(0.6):
            return 2  # Somewhat Active (Healthy)
        elif score < merged_df['CombinedHealthScore'].quantile(0.8):
            return 3  # Active (Very Healthy)
        else:
            return 4  # Highly Active (Extremely Healthy)

    # Create health status based on established scores
    merged_df['HealthStatus'] = merged_df['CombinedHealthScore'].apply(categorize_health_status)

    # Print distribution of health status
    print("\nHealth Status Distribution:")
    print(merged_df['HealthStatus'].value_counts().sort_index())

    # Prepare raw features for the model (excluding the health scores)
    raw_features = ['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes',
                    'SedentaryMinutes', 'Calories', 'total_sleep_minutes', 'HeartRate', 'BMI']

    X = merged_df[raw_features]
    y = merged_df['HealthStatus']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create training and testing datasets with all features and target
    train_data = X_train.copy()
    train_data['HealthStatus'] = y_train
    train_data['CombinedHealthScore'] = merged_df.loc[train_data.index, 'CombinedHealthScore']
    test_data = X_test.copy()
    test_data['HealthStatus'] = y_test
    test_data['CombinedHealthScore'] = merged_df.loc[test_data.index, 'CombinedHealthScore']

    # Save the datasets to CSV files
    train_data.to_csv('datasets/training_dataset.csv', index=False)
    test_data.to_csv('datasets/testing_dataset.csv', index=False)

    print("\nDatasets have been saved to:")
    print("- datasets/training_dataset.csv")
    print("- datasets/testing_dataset.csv")

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'raw_features': raw_features,
        'imputer': imputer,
        'scaler': scaler
    }

if __name__ == "__main__":
    # Run preprocessing if this file is run directly
    data = load_and_preprocess_data() 