# import joblib
# import numpy as np

# # Load the trained model
# model = joblib.load("monthly_sleep_score_model.pkl")

# def predict_sleep_score(avg_sleep_minutes):
#     """
#     Predict sleep score based on monthly average sleep minutes.
#     """
#     # Define optimal sleep thresholds
#     optimal_min, optimal_max = 420, 540  # Ideal sleep range in minutes
    
#     # Calculate deviation score
#     deviation = min(abs(avg_sleep_minutes - optimal_min), abs(avg_sleep_minutes - optimal_max))
#     deviation_score = 100 - (deviation / (optimal_max - optimal_min)) * 100
#     deviation_score = np.clip(deviation_score, 0, 100)

#     # Prepare input data for model prediction
#     input_features = np.array([[avg_sleep_minutes, deviation_score]])

#     # Predict sleep score
#     predicted_score = model.predict(input_features)[0]
    
#     return round(predicted_score, 2)

# # Example usage
# if __name__ == "__main__":
#     avg_sleep = float(input("Enter average sleep minutes for a month: "))  # User input
#     sleep_score = predict_sleep_score(avg_sleep)
#     print(f"Predicted Sleep Score for {avg_sleep} minutes: {sleep_score}")




import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("monthly_sleep_score_model.pkl")

def predict_sleep_score(avg_sleep_minutes):
    """
    Predict sleep score based on monthly average sleep minutes using the trained model.
    """
    # Define optimal sleep thresholds
    optimal_min, optimal_max = 420, 540  # Ideal sleep range in minutes
    
    # Calculate weighted deviation score
    weight_exponent = 1.5  # Adjust this exponent to control sensitivity
    deviation = min(abs(avg_sleep_minutes - optimal_min), abs(avg_sleep_minutes - optimal_max))
    weighted_deviation = deviation ** weight_exponent
    
    # Normalize deviation score
    max_weighted_dev = 120 ** weight_exponent  # Approximate max deviation
    deviation_score = 100 - (weighted_deviation / max_weighted_dev) * 100
    deviation_score = np.clip(deviation_score, 0, 100)
    
    # Prepare input data for model prediction
    input_features = np.array([[avg_sleep_minutes, deviation_score]])
    
    # Predict sleep score
    predicted_score = model.predict(input_features)[0]
    
    return round(predicted_score, 2)

# Example usage
if __name__ == "__main__":
    avg_sleep = float(input("Enter average sleep minutes for a month: "))  # User input
    sleep_score = predict_sleep_score(avg_sleep)
    print(f"Predicted Sleep Score for {avg_sleep} minutes: {sleep_score}")