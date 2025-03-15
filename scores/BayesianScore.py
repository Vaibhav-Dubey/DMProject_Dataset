import pandas as pd
import numpy as np
import py as pm
import theano.tensor as tt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Load the dataset
activity_df = pd.read_csv("dateset/Fixed_Merged_Activity_Weight.csv")
sleep_df = pd.read_csv("dateset/preprocessed_sleep_data.csv")  # Assuming sleep data is in this file
heart_rate_df = pd.read_csv("dateset/filled_heart_rate_data.csv")  # Assuming heart rate data is in this file

# Merge datasets
activity_df['ActivityDate'] = pd.to_datetime(activity_df['ActivityDate'])
sleep_df['date'] = pd.to_datetime(sleep_df['date'])
heart_rate_df['Time'] = pd.to_datetime(heart_rate_df['Time'])

# Aggregate heart rate data by day and user
heart_rate_df['date'] = heart_rate_df['Time'].dt.date
daily_heart_rate = heart_rate_df.groupby(['Id', 'date'])['Value'].mean().reset_index()
daily_heart_rate['date'] = pd.to_datetime(daily_heart_rate['date'])

# Merge all datasets on Id and date
merged_df = pd.merge(activity_df, sleep_df, how='left', left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'])
merged_df = pd.merge(merged_df, daily_heart_rate, how='left', left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'])

# Fill missing sleep and heart rate data (forward fill for each user)
merged_df['total_sleep_minutes'] = merged_df.groupby('Id')['total_sleep_minutes'].ffill()
merged_df['Value'] = merged_df.groupby('Id')['Value'].ffill()

# Drop rows with missing values
merged_df = merged_df.dropna(subset=['BMI', 'SedentaryMinutes', 'TotalSteps', 'Calories', 'total_sleep_minutes', 'Value'])

# Add a synthetic event and time-to-event column for demonstration
# In a real-world scenario, this data should come from your dataset
np.random.seed(42)
merged_df['Event'] = np.random.randint(0, 2, size=len(merged_df))  # 1 if event occurred, 0 if censored
merged_df['TimeToEvent'] = np.random.exponential(scale=100, size=len(merged_df))  # Synthetic time-to-event data

# Define covariates
covariates = ['BMI', 'SedentaryMinutes', 'TotalSteps', 'Calories', 'total_sleep_minutes', 'Value']

# Standardize covariates
merged_df[covariates] = (merged_df[covariates] - merged_df[covariates].mean()) / merged_df[covariates].std()

# Bayesian Cox Proportional Hazards Model with PyMC3
with pm.Model() as bayesian_cox_model:
    # Priors for coefficients
    beta = pm.Normal('beta', mu=0, sigma=1, shape=len(covariates))

    # Linear predictor
    X = merged_df[covariates].values
    linear_predictor = tt.dot(X, beta)

    # Likelihood (partial likelihood for Cox model)
    def log_likelihood(value):
        return -CoxPHFitter().fit(merged_df, duration_col='TimeToEvent', event_col='Event', covariates=covariates).log_likelihood_

    # Define likelihood
    pm.Potential('likelihood', log_likelihood(linear_predictor))

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, cores=2)

# Summarize the posterior distribution
pm.summary(trace)

# Evaluate the model
# Use the posterior mean of the coefficients to calculate the risk score
posterior_mean_beta = trace['beta'].mean(axis=0)
merged_df['RiskScore'] = np.dot(merged_df[covariates].values, posterior_mean_beta)

# Calculate Concordance Index (C-index)
c_index = concordance_index(merged_df['TimeToEvent'], -merged_df['RiskScore'], merged_df['Event'])
print(f"Concordance Index (C-index): {c_index}")

# Plot posterior distributions of coefficients
pm.plot_posterior(trace, var_names=['beta'])
plt.show()
