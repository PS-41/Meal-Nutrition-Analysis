import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MultimodalModel  # Import the model class
from data_prep import MultimodalDataset  # Import the dataset class
from model_training_utility import train_full_model, k_fold_cross_validation, plot_multiple_training_curves
import os

if __name__ == "__main__":
    # Load the dataset
    metadata = torch.load("../data/dataloader_metadata.pth")
    dataset = metadata["dataset"]

    # Load the best hyperparameters
    best_hyperparameters_path = "../results/best_hyperparameters.pth"
    if not os.path.exists(best_hyperparameters_path):
        raise FileNotFoundError(f"\nBest hyperparameters file not found at {best_hyperparameters_path}. Please run tune_hyperparameters.py first.")

    best_hyperparameters = torch.load(best_hyperparameters_path)
    print(f"Best Hyperparameters Loaded: {best_hyperparameters}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 20
    k_folds = 5

    # Perform K-Fold Cross-Validation using the best hyperparameters
    print("Starting K-Fold Cross-Validation...")
    fold_results = k_fold_cross_validation(
            dataset=dataset,
            hyperparams=best_hyperparameters,
            num_epochs=num_epochs,
            k_folds=k_folds,
            device=device
        )


    # Plot the training and validation loss for each fold
    plot_multiple_training_curves(fold_results, save_path="../results/combined_training_curves.png")

    # Train the final model on the full dataset
    print("Training on the full dataset with the best hyperparameters...")
    model = MultimodalModel(
                image_size=(64, 64, 3),
                cgm_size=16,
                demo_size=31,
                dropout_rate=best_hyperparameters['dropout_rate'],
                cnn_filters=best_hyperparameters['cnn_filters'],
                lstm_hidden_size=best_hyperparameters['lstm_hidden_size'],
                num_lstm_layers=best_hyperparameters['num_lstm_layers']
            )
    if best_hyperparameters['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_hyperparameters['learning_rate'], weight_decay=best_hyperparameters['weight_decay'])
    elif best_hyperparameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=best_hyperparameters['learning_rate'], momentum=0.9, weight_decay=best_hyperparameters['weight_decay'])
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=best_hyperparameters['learning_rate'], weight_decay=best_hyperparameters['weight_decay'])
    
    full_loader = DataLoader(dataset, batch_size=best_hyperparameters['batch_size'], shuffle=True)

    train_full_model(
        model=model,
        dataloader=full_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Save the final trained model
    torch.save(model.state_dict(), "../results/multimodal_model.pth")
    print("Final model trained and saved.")