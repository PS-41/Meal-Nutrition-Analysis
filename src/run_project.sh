#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Running data_prep.py..."
python data_prep.py

echo "Running tune_hyperparameters.py..."
python tune_hyperparameters.py

echo "Running train_model.py..."
python train_model.py

echo "Running predict_test_set.py..."
python predict_test_set.py

echo "All scripts executed successfully!"
