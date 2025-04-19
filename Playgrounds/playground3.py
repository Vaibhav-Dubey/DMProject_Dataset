import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv('averages_per_id.csv')

# Display basic information about the dataset
print("Columns in the dataset:", df.columns.tolist())

print(df.head())

# Define the metric columns to analyze
# Adjust these names if your CSV uses different labels (e.g., 'calories_burned' instead of 'calories')
metrics = ['totalsteps', 'calories', 'veryactivedistance', 'weightkg', 'moderatelyactivedistance', 'lightactivedistance', 'sedentaryactivedistance', 'veryactiveminutes', 'fairlyactiveminutes', 'lightlyactiveminutes', 'sedentaryminutes', 'bmi']

# Verify that each metric exists in the DataFrame
for metric in metrics:
    if metric not in df.columns:
        print(f"Warning: '{metric}' column not found in the dataset.")

# -----------------------------
# 1. Histograms for Each Metric
# -----------------------------
for metric in metrics:
    if metric in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[metric], kde=True, bins=30)
        plt.title(f'Histogram of {metric.capitalize()}')
        plt.xlabel(metric.capitalize())
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# -----------------------------
# 2. Boxplots for Each Metric
# -----------------------------
plt.figure(figsize=(12, 4))
for i, metric in enumerate(metrics):
    if metric in df.columns:
        plt.subplot(1, len(metrics), i + 1)
        sns.boxplot(y=df[metric])
        plt.title(f'Boxplot of {metric.capitalize()}')
        plt.ylabel('')
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Pairplot for Visual Relationships
# -----------------------------
# This will plot pairwise relationships between steps, weight, and calories
existing_metrics = [m for m in metrics if m in df.columns]
if existing_metrics:
    sns.pairplot(df[existing_metrics])
    plt.suptitle("Pairplot of Steps, Weight, and Calories", y=1.02)
    plt.show()

# -----------------------------
# 4. Correlation Heatmap
# -----------------------------
if existing_metrics:
    correlation = df[existing_metrics].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
