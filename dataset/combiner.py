import pandas as pd

# Load datasets
file_activity = "./Grouped_Daily_Activity.csv"
file_ds1 = "./DS1.csv"

df_activity = pd.read_csv(file_activity)
df_ds1 = pd.read_csv(file_ds1)

# Rename columns in Grouped_Daily_Activity.csv to match DS1
df_activity.rename(columns={'Id': 'id', 'ActivityDate': 'date'}, inplace=True)

# Convert date columns to string for merging (ignore time)
df_activity['date'] = df_activity['date'].astype(str)
df_ds1['date'] = df_ds1['date'].astype(str)

# Merge datasets on 'id' and 'date'
df_merged = df_ds1.merge(df_activity, on=['id', 'date'], how='outer')

# Handle duplicate column names by keeping only one version
df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

# Save to CSV
output_file = "merged_ds1_grouped_activity.csv"
df_merged.to_csv(output_file, index=False)

print(f"Merged dataset saved to {output_file}")
