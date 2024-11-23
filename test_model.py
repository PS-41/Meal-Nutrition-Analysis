import torch
from model import MultimodalModel
from data_prep import MultimodalDataset  # Import the class definition
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Step 1: Load dataset metadata
    metadata = torch.load("../data/dataloader_metadata.pth")

    # Recreate DataLoader
    dataset = metadata["dataset"]
    batch_size = metadata["batch_size"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("DataLoader loaded and reconstructed successfully.")

    # Step 2: Initialize the model
    model = MultimodalModel()

    # Step 3: Get a batch from the DataLoader and test the model
    for batch in dataloader:
        image_breakfast = batch["image_breakfast"].permute(0, 3, 1, 2)  # Change to (B, C, H, W)
        image_lunch = batch["image_lunch"].permute(0, 3, 1, 2)
        cgm_data = batch["cgm_data"]
        demo_data = batch["demo_viome_data"]

        # Forward pass
        output = model(image_breakfast, image_lunch, cgm_data, demo_data)
        print(f"Output shape: {output.shape}")
        break
