import pandas as pd

activity_file = "Grouped_Daily_Activity.csv"
weight_file = "weightLogInfo_merged.csv_combined_file.csv"

activity_df = pd.read_csv(activity_file)
weight_df = pd.read_csv(weight_file)

activity_df["ActivityDate"] = pd.to_datetime(activity_df["ActivityDate"]).dt.date
weight_df["Date"] = pd.to_datetime(weight_df["Date"]).dt.date

merged_df = pd.merge(activity_df, weight_df, how="left", left_on=["Id", "ActivityDate"], right_on=["Id", "Date"])

merged_df.drop(columns=["Date"], inplace=True)

merged_df.sort_values(by=["Id", "ActivityDate"], inplace=True)

merged_df["WeightKg"] = merged_df.groupby("Id")["WeightKg"].ffill().bfill()
merged_df["WeightPounds"] = merged_df.groupby("Id")["WeightPounds"].ffill().bfill()
merged_df["BMI"] = merged_df.groupby("Id")["BMI"].ffill().bfill()

merged_csv_path = "Fixed_Merged_Activity_Weight.csv"
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged dataset saved at: {merged_csv_path}")
print(merged_df.head())
