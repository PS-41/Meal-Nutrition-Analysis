import pandas as pd
import numpy as np

# Load the CSV files
predicted = pd.read_csv('test_predictions.csv')
actual = pd.read_csv('lol.csv')

# Extract the 'label' columns
y_pred = predicted['label'].values
y_actual = actual['label'].values

# Ensure both arrays have the same length
if len(y_pred) != len(y_actual):
    raise ValueError("The number of rows in the prediction and actual data files do not match.")

# Calculate RMSRE
rmsre = np.sqrt(np.mean(((y_pred - y_actual) / y_actual) ** 2))

# Print the result
print("Root Mean Square Relative Error (RMSRE):", rmsre)
