import numpy as np
import pandas as pd
from xg_boost_model import train_xgboost, predict_health_status

class HealthRouter:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.imputer = None
        self.scaler = None

    def initialize_model(self):
        """Train and store the specified model."""
        if self.model_type == 'xgboost':
            self.model, self.imputer, self.scaler = train_xgboost()
        # Add other models here (e.g., 'svm', 'randomforest')
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return self

    def predict(self, input_data):
        """Predict health status for input data."""
        if not all(k in input_data for k in [
            'TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 
            'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories',
            'total_sleep_minutes', 'HeartRate', 'BMI'
        ]):
            raise ValueError("Input data missing required features.")

        result = predict_health_status(self.model, self.imputer, self.scaler, input_data)
        
        return {
            'prediction': result['prediction'],
            'probabilities': dict(zip(result['status_labels'], result['probabilities']))
        }
