import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('datasets/training_dataset.csv')
test_data = pd.read_csv('datasets/testing_dataset.csv')

# Define feature columns (based on your raw_features in the preprocessing code)
raw_features = ['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes',
                'SedentaryMinutes', 'Calories', 'total_sleep_minutes', 'HeartRate', 'BMI']

# Extract features and target
X_train = train_data[raw_features]
y_train = train_data['HealthStatus']
X_test = test_data[raw_features]
y_test = test_data['HealthStatus']

# Handle any missing values (though your preprocessing already handled most)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Scale the features (KNN is distance-based, so scaling is crucial)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Start with k=5
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Sedentary', 'Low Active', 'Somewhat Active', 'Active', 'Highly Active']))

# Optional: Experiment with different k values to find the best one
k_values = range(1, 21)  # Test k from 1 to 20
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Training accuracy
    train_pred = knn.predict(X_train_scaled)
    train_scores.append(accuracy_score(y_train, train_pred))

    # Test accuracy
    test_pred = knn.predict(X_test_scaled)
    test_scores.append(accuracy_score(y_test, test_pred))

# Plot accuracy for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Training Accuracy', marker='o')
plt.plot(k_values, test_scores, label='Test Accuracy', marker='o')
plt.title('KNN Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Print the best k value based on test accuracy
best_k = k_values[np.argmax(test_scores)]
print(f"\nBest k value: {best_k} with Test Accuracy: {max(test_scores):.4f}")