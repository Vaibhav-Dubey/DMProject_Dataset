import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_preprocessed_data():
    """
    Load preprocessed data from CSV files
    """
    # Load training and testing datasets
    train_data = pd.read_csv('datasets/training_dataset.csv')
    test_data = pd.read_csv('datasets/testing_dataset.csv')
    
    # Separate features and target
    raw_features = ['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes',
                   'SedentaryMinutes', 'Calories', 'total_sleep_minutes', 'HeartRate', 'BMI']
    
    X_train = train_data[raw_features]
    y_train = train_data['HealthStatus']
    X_test = test_data[raw_features]
    y_test = test_data['HealthStatus']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'raw_features': raw_features,
        'imputer': imputer,
        'scaler': scaler
    }

def train_logistic_regression():
    """
    Train and evaluate a logistic regression model for health status prediction
    """
    # Load preprocessed data from CSV files
    data = load_preprocessed_data()
    
    # Perform cross-validation with Logistic Regression
    log_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    cv_scores = cross_val_score(log_model, data['X_train_scaled'], data['y_train'], cv=5)
    print("\nCross-validation scores:", cv_scores)
    print("Average CV score:", cv_scores.mean())
    print("CV score standard deviation:", cv_scores.std())

    # Train the final model
    log_model.fit(data['X_train_scaled'], data['y_train'])

    # Make predictions
    y_pred = log_model.predict(data['X_test_scaled'])

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(data['y_test'], y_pred, 
                              target_names=['Sedentary', 'Low Active', 'Somewhat Active', 
                                          'Active', 'Highly Active']))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(data['y_test'], y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sedentary', 'Low Active', 'Somewhat Active', 'Active', 'Highly Active'],
                yticklabels=['Sedentary', 'Low Active', 'Somewhat Active', 'Active', 'Highly Active'])
    plt.title('Confusion Matrix - Logistic Regression')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': data['raw_features'],
        'importance': np.abs(log_model.coef_).mean(axis=0)  # Average absolute coefficient across all classes
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance - Logistic Regression')
    plt.tight_layout()
    plt.show()

    return log_model, data['imputer'], data['scaler']

def predict_health_status(model, imputer, scaler, new_data):
    """
    Predict health status for new data using raw features
    new_data should be a dictionary with the same raw features as used in training
    """
    # Convert new data to DataFrame
    new_df = pd.DataFrame([new_data])
    
    # Handle missing values
    new_imputed = imputer.transform(new_df)
    
    # Scale the features
    new_scaled = scaler.transform(new_imputed)
    
    # Make prediction
    prediction = model.predict(new_scaled)[0]
    
    # Get probability estimates
    probabilities = model.predict_proba(new_scaled)[0]
    
    # Convert prediction to health status
    health_statuses = ['Sedentary (Unhealthy)', 'Low Active (Moderately Healthy)', 
                      'Somewhat Active (Healthy)', 'Active (Very Healthy)', 
                      'Highly Active (Extremely Healthy)']
    
    return health_statuses[prediction], probabilities

if __name__ == "__main__":
    # Train the model and get the trained model and preprocessing objects
    model, imputer, scaler = train_logistic_regression()
    
    # Example usage
    example_data = {
        'TotalSteps': 8000,
        'VeryActiveMinutes': 30,
        'FairlyActiveMinutes': 45,
        'LightlyActiveMinutes': 120,
        'SedentaryMinutes': 300,
        'Calories': 2000,
        'total_sleep_minutes': 480,
        'HeartRate': 70,
        'BMI': 25
    }

    print("\nExample Prediction:")
    status, probabilities = predict_health_status(model, imputer, scaler, example_data)
    print(f"Predicted Health Status: {status}")
    print("\nProbability estimates for each class:")
    for i, prob in enumerate(probabilities):
        print(f"{['Sedentary', 'Low Active', 'Somewhat Active', 'Active', 'Highly Active'][i]}: {prob:.2%}") 