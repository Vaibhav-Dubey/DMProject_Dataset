import pandas as pd

# Load the final merged dataset
file_path = "./final_merged_health_data.csv"
df = pd.read_csv(file_path)

# Convert the 'date' column to a uniform date format while handling mixed formats
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce').dt.date

# Save the preprocessed file
processed_file_path = "./final_merged_health_data_preprocessed.csv"
df.to_csv(processed_file_path, index=False)

# Confirm processing completion
processed_file_path
