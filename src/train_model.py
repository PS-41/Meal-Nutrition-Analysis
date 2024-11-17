import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MultimodalModel  # Import the model class
from data_prep import MultimodalDataset  # Import the dataset class

# Define RMSRE Loss Function
class RMSRELoss(nn.Module):
    def __init__(self):
        super(RMSRELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # To prevent division by zero
        relative_error = (y_true - y_pred) / (y_true + epsilon)
        return torch.sqrt(torch.mean(relative_error**2))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)
    loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    return loss_history, val_loss_history

def train_full_model(model, dataloader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Full)"):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Full Dataset Loss (RMSRE): {epoch_loss:.4f}")

    return loss_history

def plot_training_curve(train_loss, val_loss, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label="Train Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='o', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("RMSRE Loss")
    plt.title("Training and Validation Curve")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    metadata = torch.load("../data/dataloader_metadata.pth")
    dataset = metadata["dataset"]
    batch_size = metadata["batch_size"]

    val_split = 0.2
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    print("Train and Validation DataLoaders created successfully.")

    model = MultimodalModel(image_size=(64, 64, 3), cgm_size=16, demo_size=54)
    criterion = RMSRELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    device = 'cpu'
    train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    plot_training_curve(train_loss, val_loss, save_path="../results/training_curve.png")

    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Training on the full dataset now...")
    full_loss_history = train_full_model(model, full_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    torch.save(model.state_dict(), "../results/final_multimodal_model.pth")
    print("Model training on full dataset complete and saved.")
