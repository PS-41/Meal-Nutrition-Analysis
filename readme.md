# Project Workflow

This project consists of multiple steps that need to be executed in sequence for the entire pipeline to work. All codes files are present in src folder. Below are the steps:

1. **Data Preparation** (`data_prep.py`): Prepares and processes the data for training and testing and saves the metadata in data folder.
2. **Hyperparameter Tuning** (`tune_hyperparameters.py`): Optimizes the hyperparameters for the model and saves the optimal hyperparameter configuration in results folder.
3. **Model Training** (`train_model.py`): Trains the machine learning model using the prepared data and optimal hyperparameters and saves the model in results folder.
4. **Prediction on Test Set** (`predict_test_set.py`): Generates predictions for the test dataset using the trained model and saves the predictions in the results folder.

## Running the Project

To execute the entire project in one go, follow these steps:

1. Ensure you have Python installed on your machine.
2. Make sure all the dependencies are installed.
3. Either execute each of the above python files one by one or run the script using command: ./run_project.sh