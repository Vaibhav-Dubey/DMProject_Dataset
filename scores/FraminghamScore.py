import pandas as pd
import matplotlib.pyplot as plt
import os

# Load datasets
activity_df = pd.read_csv("dateset/Fixed_Merged_Activity_Weight.csv")
sleep_df = pd.read_csv("dateset/preprocessed_sleep_data.csv")  # Assuming sleep data is in this file
heart_rate_df = pd.read_csv("dateset/filled_heart_rate_data.csv")  # Assuming heart rate data is in this file

# Convert date columns to datetime
activity_df['ActivityDate'] = pd.to_datetime(activity_df['ActivityDate'])
sleep_df['date'] = pd.to_datetime(sleep_df['date'])
heart_rate_df['Time'] = pd.to_datetime(heart_rate_df['Time'])

# Aggregate heart rate data by day and user
heart_rate_df['date'] = heart_rate_df['Time'].dt.date
daily_heart_rate = heart_rate_df.groupby(['Id', 'date'])['Value'].mean().reset_index()
daily_heart_rate['date'] = pd.to_datetime(daily_heart_rate['date'])

# Merge all datasets on Id and date
merged_df = pd.merge(activity_df, sleep_df, how='left', left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'])
merged_df = pd.merge(merged_df, daily_heart_rate, how='left', left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'])

# Fill missing sleep and heart rate data (forward fill for each user)
merged_df['total_sleep_minutes'] = merged_df.groupby('Id')['total_sleep_minutes'].ffill()
merged_df['Value'] = merged_df.groupby('Id')['Value'].ffill()

# Define weights for HRS
weights = {
    'BMI': 0.3,
    'Sedentary': 0.2,
    'Steps': 0.2,
    'Calories': 0.1,
    'Sleep': 0.1,
    'HeartRate': 0.1
}

# Calculate HRS
merged_df['HRS'] = (
    weights['BMI'] * merged_df['BMI'] +
    weights['Sedentary'] * (merged_df['SedentaryMinutes'] / 1000) +
    weights['Steps'] * (merged_df['TotalSteps'] / 10000) +
    weights['Calories'] * (merged_df['Calories'] / 2000) +
    weights['Sleep'] * (merged_df['total_sleep_minutes'] / 480) +  # 480 minutes = 8 hours (ideal sleep)
    weights['HeartRate'] * (merged_df['Value'] / 60)  # Normalize heart rate (e.g., 60 bpm = baseline)
)

merged_df['FraminghamRiskScore'] = (
    merged_df['BMI'] +
    (merged_df['SedentaryMinutes'] / 1000) -
    (merged_df['TotalSteps'] / 10000) +
    (merged_df['total_sleep_minutes'] / 480) -  # Sleep contributes positively to health
    (merged_df['Value'] / 60)  # Higher heart rate increases risk
)

# Calculate PAI
merged_df['PAI'] = (merged_df['VeryActiveMinutes'] * 2) + (merged_df['FairlyActiveMinutes'] * 1.5) + (merged_df['LightlyActiveMinutes'] * 1)

# Group by ID
grouped = merged_df.groupby('Id')


for id, group in grouped:
    # Plot Framingham Risk Score
    plt.figure(figsize=(10, 5))
    plt.plot(group['ActivityDate'], group['FraminghamRiskScore'], marker='o', color='red', label='Framingham Risk Score')
    plt.xlabel('Date')
    plt.ylabel('Framingham Risk Score')
    plt.title(f'Framingham Risk Score Over Time for ID: {id}')
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/ID_{id}/Framingham_Risk_Score_ID_{id}.png', dpi=300, bbox_inches='tight')
    plt.show()
