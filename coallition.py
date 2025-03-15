import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv('Fixed_Merged_Activity_Weight.csv')

# Normalize column names to lower case for consistency
df.columns = df.columns.str.lower()

# Print the first few rows and the available columns to inspect the data structure
print("Columns in the dataset:")
print(df.columns.tolist())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# # Define the required columns (adjust if your CSV uses slightly different names)
# # We're looking for 'id', 'steps', 'weight', and 'calories burned'
required_columns = [['id', 'activitydate', 'totalsteps', 'totaldistance', 'trackerdistance', 'loggedactivitiesdistance', 'veryactivedistance', 'moderatelyactivedistance', 'lightactivedistance', 'sedentaryactivedistance', 'veryactiveminutes', 'fairlyactiveminutes', 'lightlyactiveminutes', 'sedentaryminutes', 'calories', 'weightkg', 'fat', 'bmi']
]

# # For calories burned, check for common variations in naming
# if 'calories burned' in df.columns:
#     calories_col = 'calories burned'
# elif 'calories_burned' in df.columns:
#     calories_col = 'calories_burned'
# elif 'calories' in df.columns:
#     calories_col = 'calories'
# else:
#     calories_col = None

# if calories_col is not None:
#     required_columns.append(calories_col)
# else:
#     print("No column found for 'calories burned' or similar. Please check your CSV column names.")

# # Verify that all required columns exist in the DataFrame
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     print(f"Missing columns in CSV: {missing_columns}")
# else:
#     # Group the DataFrame by 'id' and compute the mean for steps, weight, and calories burned
grouped = df.groupby('id')[[ 'totalsteps', 'calories', 'veryactivedistance' , 'weightkg','moderatelyactivedistance', 'lightactivedistance', 'sedentaryactivedistance', 'veryactiveminutes', 'fairlyactiveminutes', 'lightlyactiveminutes', 'sedentaryminutes','bmi']].mean().reset_index()
print("\nAverage metrics for each ID:")
print(grouped)
    
#     # Save the grouped averages to a new CSV file (optional)
grouped.to_csv('averages_per_id.csv', index=False)
print("\nAverages saved to 'averages_per_id.csv'.")

#     # Calculate and print the correlation matrix for these metrics
#     correlation_matrix = grouped[['steps', 'weight', calories_col]].corr()
#     print("\nCorrelation Matrix:")
#     print(correlation_matrix)

#     # Visualize the correlation matrix using a heatmap
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#     plt.title("Correlation between Steps, Weight, and Calories Burned per ID")
#     plt.tight_layout()
#     plt.show()

#     # Plot a pairplot (scatterplot matrix) for the grouped metrics
#     sns.pairplot(grouped[['steps', 'weight', calories_col]])
#     plt.suptitle("Pairplot for Steps, Weight, and Calories Burned per ID", y=1.02)
#     plt.show()
