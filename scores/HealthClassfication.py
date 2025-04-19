import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
activity_df = pd.read_csv("dateset/Fixed_Merged_Activity_Weight.csv")

# Convert ActivityDate to datetime
activity_df['ActivityDate'] = pd.to_datetime(activity_df['ActivityDate'])

# Group by ID and calculate average daily steps
average_steps = activity_df.groupby('Id')['TotalSteps'].mean().reset_index()

# Function to categorize health status based on step count
def categorize_health_status(steps):
    if steps < 5000:
        return "Sedentary (Unhealthy)"
    elif 5000 <= steps < 7500:
        return "Low Active (Moderately Healthy)"
    elif 7500 <= steps < 10000:
        return "Somewhat Active (Healthy)"
    elif 10000 <= steps < 12500:
        return "Active (Very Healthy)"
    else:
        return "Highly Active (Extremely Healthy)"

# Apply health status categorization
average_steps['HealthStatus'] = average_steps['TotalSteps'].apply(categorize_health_status)

# Create a scatter plot
plt.figure(figsize=(12, 6))

# Define colors for each health status
colors = {
    "Sedentary (Unhealthy)": "red",
    "Low Active (Moderately Healthy)": "orange",
    "Somewhat Active (Healthy)": "yellow",
    "Active (Very Healthy)": "green",
    "Highly Active (Extremely Healthy)": "darkgreen"
}

# Plot each point with the corresponding color
for status, color in colors.items():
    subset = average_steps[average_steps['HealthStatus'] == status]
    plt.scatter(subset['Id'], subset['TotalSteps'], color=color, label=status)

# Add labels, title, and legend
plt.xlabel('User ID')
plt.ylabel('Average Daily Steps')
plt.title('Average Daily Steps and Health Status by User ID')
plt.legend(title='Health Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
