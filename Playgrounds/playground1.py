import pandas as pd

file_path = "DailyActivity.csv"
df = pd.read_csv(file_path)

df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

grouped_df = df.groupby(["Id", "ActivityDate"]).agg({
    "TotalSteps": "sum",
    "TotalDistance": "sum",
    "TrackerDistance": "sum",
    "LoggedActivitiesDistance": "sum",
    "VeryActiveDistance": "sum",
    "ModeratelyActiveDistance": "sum",
    "LightActiveDistance": "sum",
    "SedentaryActiveDistance": "sum",
    "VeryActiveMinutes": "sum",
    "FairlyActiveMinutes": "sum",
    "LightlyActiveMinutes": "sum",
    "SedentaryMinutes": "sum",
    "Calories": "sum",
}).reset_index()

grouped_csv_path = "Grouped_Daily_Activity.csv"
grouped_df.to_csv(grouped_csv_path, index=False)

individual_analysis = grouped_df.groupby("Id").agg({
    "TotalSteps": ["mean", "max", "min"],
    "TotalDistance": ["mean", "max", "min"],
    "VeryActiveMinutes": ["mean", "max", "min"],
    "FairlyActiveMinutes": ["mean", "max", "min"],
    "LightlyActiveMinutes": ["mean", "max", "min"],
    "SedentaryMinutes": ["mean", "max", "min"],
    "Calories": ["mean", "max", "min"],
}).reset_index()

analysis_csv_path = "Individual_Activity_Analysis.csv"
individual_analysis.to_csv(analysis_csv_path, index=False)

print(f"Grouped Daily Activity saved at: {grouped_csv_path}")
print(f"Individual Activity Analysis saved at: {analysis_csv_path}")
