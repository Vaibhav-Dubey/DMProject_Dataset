# Fill missing values with the median per user id
import pandas as pd

# Load the dataset
file_path = ".cd/merged_ds1_grouped_activity.csv"
df = pd.read_csv(file_path)
df_filled_per_user = df.groupby("id").apply(lambda group: group.fillna(group.median(numeric_only=True))).reset_index(drop=True)

# Save the processed file
processed_file_path_per_user = "./processed_ds1_grouped_activity_per_user.csv"
df_filled_per_user.to_csv(processed_file_path_per_user, index=False)

