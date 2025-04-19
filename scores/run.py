# run.py
from health_router import HealthRouter

# Example input data (multiple samples)
input_samples = [
    {  # Sample 1
        'TotalSteps': 8000,
        'VeryActiveMinutes': 30,
        'FairlyActiveMinutes': 45,
        'LightlyActiveMinutes': 120,
        'SedentaryMinutes': 300,
        'Calories': 2000,
        'total_sleep_minutes': 480,
        'HeartRate': 70,
        'BMI': 25
    },
    {  # Sample 2
        'TotalSteps': 5000,
        'VeryActiveMinutes': 10,
        'FairlyActiveMinutes': 20,
        'LightlyActiveMinutes': 200,
        'SedentaryMinutes': 600,
        'Calories': 1800,
        'total_sleep_minutes': 400,
        'HeartRate': 75,
        'BMI': 28
    }
]

# Initialize router and get predictions
router = HealthRouter(model_type='xgboost').initialize_model()

for input_data in input_samples:
    predictions = router.predict(input_data)

    # Display results
    for idx, pred in enumerate(predictions):
        print(f"\n--- Prediction for Sample {idx + 1} ---")
        print("Input Features:", pred["input_data"])
        print("Predicted Health Status:", pred["prediction"])
        print("Probabilities:")
        for i, prob in enumerate(pred["probabilities"]):
            print(f"  {['Sedentary', 'Low Active', 'Somewhat Active', 'Active', 'Highly Active'][i]}: {prob:.2%}")
