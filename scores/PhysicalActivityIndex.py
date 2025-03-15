import pandas as pd
import matplotlib.pyplot as plt
import os

# Load datasets
activity_df = pd.read_csv("dateset/Fixed_Merged_Activity_Weight.csv")
sleep_df = pd.read_csv("dateset/preprocessed_sleep_data.csv")
heart_rate_df = pd.read_csv("dateset/filled_heart_rate_data.csv")

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


# Calculate PAI
merged_df['PAI'] = (merged_df['VeryActiveMinutes'] * 2) + (merged_df['FairlyActiveMinutes'] * 1.5) + (merged_df['LightlyActiveMinutes'] * 1)

# Group by ID
grouped = merged_df.groupby('Id')

# Loop through each individual
for id, group in grouped:
    # Create a directory to save plots
    if not os.path.exists(f'plots/ID_{id}'):
        os.makedirs(f'plots/ID_{id}')

    # Create a combined plot for all metrics
    plt.figure(figsize=(12, 6))

    # Plot PAI
    plt.plot(group['ActivityDate'], group['PAI'], marker='o', color='purple', label='PAI')

    # Plot VeryActiveMinutes
    plt.plot(group['ActivityDate'], group['VeryActiveMinutes'], marker='o', color='blue', label='Very Active Minutes')

    # Plot FairlyActiveMinutes
    plt.plot(group['ActivityDate'], group['FairlyActiveMinutes'], marker='o', color='green', label='Fairly Active Minutes')

    # Plot LightlyActiveMinutes
    plt.plot(group['ActivityDate'], group['LightlyActiveMinutes'], marker='o', color='orange', label='Lightly Active Minutes')


    # Add labels, title, and legend
    plt.xlabel('Date')
    plt.ylabel('Metrics')
    plt.title(f'Activity Metrics, PAI, and Risk Scores Over Time for ID: {id}')
    plt.legend(bbox_to_anchor=(.95, .95), loc='upper left')
    plt.grid()

    # Show the plot
    plt.show()
