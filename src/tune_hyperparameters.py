import torch
from sklearn.model_selection import ParameterGrid
from model_training_utility import k_fold_cross_validation
from model import MultimodalModel
from data_prep import MultimodalDataset
from preprocessing import merge_modalities
import pandas as pd

def hyperparameter_tuning(dataset, hyperparameter_grid, num_epochs=20, k_folds=5, device='cpu'):
    """
    Perform hyperparameter tuning using k-fold cross-validation.

    Parameters:
        dataset: The dataset to use for cross-validation.
        hyperparameter_grid: A list of dictionaries, each containing a set of hyperparameters to test.
        num_folds: Number of folds for cross-validation.
        num_epochs: Number of epochs to train each model.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        dict: The best hyperparameter combination and its validation loss.
    """
    best_params = None
    best_val_loss = float("inf")

    for hyperparams in hyperparameter_grid:
        print(f"Testing hyperparameters: {hyperparams}")

        fold_results = k_fold_cross_validation(
            dataset=dataset,
            hyperparams=hyperparams,
            num_epochs=num_epochs,
            k_folds=k_folds,
            device=device
        )

        avg_val_loss = torch.mean(torch.tensor([result[1][-1] for result in fold_results]))
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = hyperparams

    return best_params, best_val_loss

if __name__ == "__main__":
    # Load the dataset
    metadata = torch.load("../data/dataloader_metadata.pth")
    dataset = metadata["dataset"]

    # Step 2: Define hyperparameter grid
    hyperparameter_grid = [
        {
            'learning_rate': lr,
            'batch_size': bs,
            'dropout_rate': dr,
            'cnn_filters': cnn_filters,
            'lstm_hidden_size': lstm_hs,
            'num_lstm_layers': num_lstm_layers,
            'weight_decay': wd,
            'optimizer': opt
        }
        for lr in [1e-3]
        for bs in [16]
        for dr in [0.2]
        for cnn_filters in [(16, 32)]
        for lstm_hs in [64]
        for num_lstm_layers in [1]
        for wd in [1e-5]
        for opt in [torch.optim.Adam]
    ]

    num_epochs = 20
    k_folds = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Perform hyperparameter tuning
    best_params, best_val_loss = hyperparameter_tuning(dataset, hyperparameter_grid, num_epochs=20, k_folds=5, device=device)
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Save the best parameters
    torch.save(best_params, "../results/best_hyperparameters.pth")
    print("Best hyperparameters saved.")
