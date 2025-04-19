import pandas as pd

# Load the datasets
df_heart_rate = pd.read_csv("./Daily_Heart_Rate_All_IDs.csv")
df_sleep = pd.read_csv("./preprocessed_sleep_data.csv")

# Standardize column names
df_heart_rate.rename(columns={'Id': 'id', 'Time': 'date'}, inplace=True)
df_sleep.rename(columns={'Id': 'id', 'date': 'date'}, inplace=True)

# Ensure 'id' and 'date' are in the correct format
df_heart_rate['date'] = pd.to_datetime(df_heart_rate['date'])
df_sleep['date'] = pd.to_datetime(df_sleep['date'])

# Merge datasets on 'id' and 'date'
merged_df = df_heart_rate.merge(df_sleep, on=['id', 'date'], how='outer')

# Save merged file
merged_df.to_csv("./merged_health_data.csv", index=False)

# Display merged file info
print("Merged dataset shape:", merged_df.shape)