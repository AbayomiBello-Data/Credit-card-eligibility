import pandas as pd
from sklearn.utils import resample

# Load the actual dataset
actual_data = pd.read_csv('data/fraud_data.csv')

# Separate the data into fraudulent and non-fraudulent transactions
fraudulent_data = actual_data[actual_data['Status'] == 1]
non_fraudulent_data = actual_data[actual_data['Status'] == 0]

# Define the number of simulated data points you want to generate
n_simulated_data_points = 100000  # Example: 100,000

# Calculate the number of fraudulent and non-fraudulent transactions to generate
n_fraudulent_transactions = int(n_simulated_data_points * actual_data['Status'].mean())
n_non_fraudulent_transactions = n_simulated_data_points - n_fraudulent_transactions

# Generate simulated fraudulent transactions by sampling with replacement
simulated_fraudulent_data = resample(fraudulent_data,
                                     replace=True,
                                     n_samples=n_fraudulent_transactions,
                                     random_state=42)

# Generate simulated non-fraudulent transactions by sampling with replacement
simulated_non_fraudulent_data = resample(non_fraudulent_data,
                                         replace=True,
                                         n_samples=n_non_fraudulent_transactions,
                                         random_state=42)

# Combine the simulated fraudulent and non-fraudulent transactions
simulated_data = pd.concat([simulated_fraudulent_data, simulated_non_fraudulent_data], ignore_index=True)

# Shuffle the simulated data
simulated_data = simulated_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the simulated data to a new file
simulated_data.to_csv('data/simulated_data.csv', index=False)