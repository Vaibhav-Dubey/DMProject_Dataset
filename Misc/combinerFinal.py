import pandas as pd

# Load the datasets
merged_health_data = pd.read_csv("./merged_health_data.csv")
df_weight = pd.read_csv("./weightLogInfo_merged.csv_combined_file.csv")

# Standardize column names
df_weight.rename(columns={'Id': 'id', 'Date': 'date'}, inplace=True)

# Convert 'date' columns to string format for merging without datetime type
merged_health_data['date'] = merged_health_data['date'].astype(str)
df_weight['date'] = df_weight['date'].astype(str)

# Merge the previously merged health data with weight dataset
final_merged_df = merged_health_data.merge(df_weight, on=['id', 'date'], how='outer')

# Save final merged file
final_merged_df.to_csv("./final_merged_health_data.csv", index=False)

# Display merged file info
print("Final merged dataset shape:", final_merged_df.shape)