import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
from model import MultimodalModel
import matplotlib.pyplot as plt
import numpy as np

# Define RMSRE Loss Function
class RMSRELoss(torch.nn.Module):
    def __init__(self):
        super(RMSRELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8
        relative_error = (y_true - y_pred) / (y_true + epsilon)
        return torch.sqrt(torch.mean(relative_error**2))


def train_model(model, train_loader, val_loader, optimizer, criterion=RMSRELoss(), num_epochs=20, device='cuda'):
    """
    Train the model and validate on a single train-validation split.

    Parameters:
        model (nn.Module): The multimodal model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module, optional): Loss function. Default is RMSRELoss().
        num_epochs (int, optional): Number of training epochs. Default is 20.
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        list: Training loss history for each epoch.
        list: Validation loss history for each epoch.
    """
    model.to(device)
    loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        tqdm_disable = (epoch + 1) % 10 != 0
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=tqdm_disable):
            image_breakfast = batch["image_breakfast"].permute(0, 3, 1, 2).to(device)
            image_lunch = batch["image_lunch"].permute(0, 3, 1, 2).to(device)
            cgm_data = batch["cgm_data"].to(device)
            demo_data = batch["demo_viome_data"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            outputs = model(image_breakfast, image_lunch, cgm_data, demo_data).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image_breakfast = batch["image_breakfast"].permute(0, 3, 1, 2).to(device)
                image_lunch = batch["image_lunch"].permute(0, 3, 1, 2).to(device)
                cgm_data = batch["cgm_data"].to(device)
                demo_data = batch["demo_viome_data"].to(device)
                labels = batch["label"].to(device).float()

                outputs = model(image_breakfast, image_lunch, cgm_data, demo_data).squeeze(1)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item() * len(labels)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    return loss_history, val_loss_history


def k_fold_cross_validation(dataset, hyperparams, num_epochs, k_folds, device):
    """
    Perform k-fold cross-validation and return training and validation losses for each fold.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    criterion = RMSRELoss()
    demo_size = len(dataset.demo_viome_data.columns)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{k_folds}")

        train_loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], sampler=SubsetRandomSampler(val_idx))

        model = MultimodalModel(
                image_size=(64, 64, 3),
                cgm_size=16,
                demo_size=demo_size,
                dropout_rate=hyperparams['dropout_rate'],
                cnn_filters=hyperparams['cnn_filters'],
                lstm_hidden_size=hyperparams['lstm_hidden_size'],
                num_lstm_layers=hyperparams['num_lstm_layers']
            )
        
        if hyperparams['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
        elif hyperparams['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning_rate'], momentum=0.9, weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])

        # Train the model for the current fold
        train_loss, val_loss = train_model(
            model, train_loader, val_loader, optimizer, criterion, num_epochs, device
        )
        fold_results.append((train_loss, val_loss))
        break

    return fold_results


def train_full_model(model, dataloader, optimizer, criterion=RMSRELoss(), num_epochs=20, device='cuda'):
    """
    Train the model on the entire dataset.
    """
    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        tqdm_disable = (epoch + 1) % 10 != 0
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Full Training)", disable=tqdm_disable):
            image_breakfast = batch["image_breakfast"].permute(0, 3, 1, 2).to(device)
            image_lunch = batch["image_lunch"].permute(0, 3, 1, 2).to(device)
            cgm_data = batch["cgm_data"].to(device)
            demo_data = batch["demo_viome_data"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            outputs = model(image_breakfast, image_lunch, cgm_data, demo_data).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Full Dataset Loss: {epoch_loss:.4f}")

    return loss_history

def plot_training_curve(train_loss, save_path=None):
    """
    Plots the training loss curve.

    Parameters:
    - train_loss (list): List of training loss values for each epoch.
    - save_path (str, optional): Path to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("RMSRE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_multiple_training_curves(fold_results, save_path=None):
    """
    Plots training and validation loss curves for multiple folds in a grid layout.

    Parameters:
    - fold_results (list of tuples): List where each element is a tuple containing
      'train_loss' and 'val_loss' for each fold.
    - save_path (str, optional): Path to save the plot. If None, the plot will not be saved.
    """
    num_folds = len(fold_results)
    cols = 3
    rows = (num_folds // cols) + (num_folds % cols > 0)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for fold_idx, (train_loss, val_loss) in enumerate(fold_results):
        ax = axes[fold_idx]
        ax.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label="Train Loss")
        ax.plot(range(1, len(val_loss) + 1), val_loss, marker='o', label="Validation Loss")
        ax.set_title(f"Fold {fold_idx + 1}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    # Remove any unused subplots (if the grid layout has extra axes)
    for i in range(len(fold_results), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
