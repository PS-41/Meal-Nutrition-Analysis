# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, SubsetRandomSampler
# from sklearn.model_selection import train_test_split
# from itertools import product
# from tqdm import tqdm
# import pandas as pd
# from model import MultimodalModel
# from train_model import RMSRELoss, train_model
# from predict_test_set import predict_test_set
# from data_prep import MultimodalDataset
# from preprocessing import merge_modalities

# def grid_search_hyperparameter_tuning(params, train_loader, val_loader, device):
#     best_model = None
#     best_params = None
#     best_val_rmsre = float('inf')

#     for lr, batch_size, num_epochs in tqdm(params, desc="Grid Search Progress"):
#         print(f"Testing hyperparameters: LR={lr}, Batch Size={batch_size}, Epochs={num_epochs}")

#         # Re-initialize model, criterion, and optimizer for each combination
#         model = MultimodalModel(image_size=(64, 64, 3), cgm_size=16, demo_size=52).to(device)
#         criterion = RMSRELoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr)

#         # Train model with current hyperparameters
#         train_loss, val_loss = train_model(
#             model, train_loader, val_loader,
#             criterion, optimizer, num_epochs=num_epochs, device=device
#         )

#         val_rmsre = val_loss[-1]
#         print(f"Validation RMSRE: {val_rmsre:.4f}")

#         # Update the best model if the validation loss is improved
#         if val_rmsre < best_val_rmsre:
#             best_val_rmsre = val_rmsre
#             best_model = model
#             best_params = {'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}

#     return best_model, best_params, best_val_rmsre

# if __name__ == "__main__":
#     # Load dataset metadata
#     metadata = torch.load("../data/dataloader_metadata.pth")
#     dataset = metadata["dataset"]
#     val_split = 0.2
#     indices = list(range(len(dataset)))
#     train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)

#     # Hyperparameter ranges for grid search
#     lr_range = [0.0001, 0.001, 0.01]
#     batch_size_range = [16, 32, 64]
#     num_epochs_range = [10, 20]
#     param_combinations = list(product(lr_range, batch_size_range, num_epochs_range))

#     # Perform grid search
#     device = 'cpu'  # Use 'cuda' if available
#     best_model, best_params, best_val_rmsre = grid_search_hyperparameter_tuning(
#         param_combinations,
#         DataLoader(dataset, batch_size=metadata["batch_size"], sampler=SubsetRandomSampler(train_indices)),
#         DataLoader(dataset, batch_size=metadata["batch_size"], sampler=SubsetRandomSampler(val_indices)),
#         device
#     )

#     print(f"Best Hyperparameters: {best_params}")
#     print(f"Best Validation RMSRE: {best_val_rmsre:.4f}")

#     # Test dataset predictions using the best model
#     img_test_path = "../data/img_test.csv"
#     cgm_test_path = "../data/cgm_test.csv"
#     viome_test_path = "../data/demo_viome_test.csv"
#     label_test_path = "../data/label_test_breakfast_only.csv"  # Optional, if needed for row_id

#     # Prepare test data
#     test_merged_data = merge_modalities(img_test_path, cgm_test_path, viome_test_path, label_test_path)
#     test_dataset = MultimodalDataset(test_merged_data, True)
#     test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

#     predictions = predict_test_set(best_model, test_loader, device=device)

#     # Save test set predictions
#     row_ids = test_merged_data.index
#     submission_df = pd.DataFrame({
#         "row_id": row_ids,
#         "label": predictions
#     })

#     submission_file_path = "../results/test_predictions_tuned.csv"
#     submission_df.to_csv(submission_file_path, index=False)
#     print(f"Test set predictions saved to {submission_file_path}.")









import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm
import pandas as pd
from model import MultimodalModel
from train_model import RMSRELoss, train_model
from predict_test_set import predict_test_set
from data_prep import MultimodalDataset
from preprocessing import merge_modalities

def grid_search_hyperparameter_tuning(params, train_loader, val_loader, device):
    best_model = None
    best_params = None
    best_val_rmsre = float('inf')

    for lr, batch_size, num_epochs, optimizer_type, weight_decay in tqdm(params, desc="Grid Search Progress"):
        print(f"Testing hyperparameters: LR={lr}, Batch Size={batch_size}, Epochs={num_epochs}, Optimizer={optimizer_type}, Weight Decay={weight_decay}")

        # Re-initialize model, criterion, and optimizer for each combination
        model = MultimodalModel(image_size=(64, 64, 3), cgm_size=16, demo_size=52).to(device)
        criterion = RMSRELoss()

        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)  # Default to Adam if none of the options match

        # Train model with current hyperparameters
        train_loss, val_loss = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, num_epochs=num_epochs, device=device
        )

        val_rmsre = val_loss[-1]
        print(f"Validation RMSRE: {val_rmsre:.4f}")

        # Update the best model if the validation loss is improved
        if val_rmsre < best_val_rmsre:
            best_val_rmsre = val_rmsre
            best_model = model
            best_params = {'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs, 'optimizer': optimizer_type, 'weight_decay': weight_decay}

    return best_model, best_params, best_val_rmsre

if __name__ == "__main__":
    # Load dataset metadata
    metadata = torch.load("../data/dataloader_metadata.pth")
    dataset = metadata["dataset"]
    val_split = 0.2
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)

    # Hyperparameter ranges for grid search
    lr_range = [0.0001, 0.001, 0.01]
    batch_size_range = [16, 32, 64]
    num_epochs_range = [10, 20]
    optimizer_range = ['adam', 'sgd', 'adamw']
    weight_decay_range = [0, 1e-4, 1e-3]
    param_combinations = list(product(lr_range, batch_size_range, num_epochs_range, optimizer_range, weight_decay_range))

    # Perform grid search
    device = 'cpu'  # Use 'cuda' if available
    best_model, best_params, best_val_rmsre = grid_search_hyperparameter_tuning(
        param_combinations,
        DataLoader(dataset, batch_size=metadata["batch_size"], sampler=SubsetRandomSampler(train_indices)),
        DataLoader(dataset, batch_size=metadata["batch_size"], sampler=SubsetRandomSampler(val_indices)),
        device
    )

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation RMSRE: {best_val_rmsre:.4f}")

    # Test dataset predictions using the best model
    img_test_path = "../data/img_test.csv"
    cgm_test_path = "../data/cgm_test.csv"
    viome_test_path = "../data/demo_viome_test.csv"
    label_test_path = "../data/label_test_breakfast_only.csv"  # Optional, if needed for row_id

    # Prepare test data
    test_merged_data = merge_modalities(img_test_path, cgm_test_path, viome_test_path, label_test_path)
    test_dataset = MultimodalDataset(test_merged_data, True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    predictions = predict_test_set(best_model, test_loader, device=device)

    # Save test set predictions
    row_ids = test_merged_data.index
    submission_df = pd.DataFrame({
        "row_id": row_ids,
        "label": predictions
    })

    submission_file_path = "../results/test_predictions_tuned.csv"
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Test set predictions saved to {submission_file_path}.")
