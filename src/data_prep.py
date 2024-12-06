import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import (
    merge_modalities,
    preprocess_images,
    preprocess_cgm,
    preprocess_demo_viome
)

class MultimodalDataset(Dataset):
    def __init__(self, merged_data, is_test=False):
        """
        Initialize the MultimodalDataset.
        
        Parameters:
            merged_data (pd.DataFrame): Merged and preprocessed dataset.
            is_test (bool): Indicates whether the dataset is a test set (no labels).
        """
        self.images_breakfast = preprocess_images(merged_data["Image Before Breakfast"])
        self.images_lunch = preprocess_images(merged_data["Image Before Lunch"])
        self.cgm_data = preprocess_cgm(merged_data["CGM Data"])
        self.demo_viome_data = preprocess_demo_viome(merged_data, is_test=is_test)
        self.is_test = is_test

        if not is_test:
            # Target variable: Lunch Calories
            self.labels = merged_data["Lunch Calories"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.cgm_data)

    def __getitem__(self, idx):
        demo_viome_row = self.demo_viome_data.iloc[idx]
        item = {
            "image_breakfast": torch.tensor(self.images_breakfast[idx], dtype=torch.float32),
            "image_lunch": torch.tensor(self.images_lunch[idx], dtype=torch.float32),
            "cgm_data": torch.tensor(self.cgm_data[idx], dtype=torch.float32),
            "demo_viome_data": torch.tensor(demo_viome_row.values, dtype=torch.float32)
        }
        if not self.is_test:
            # Add labels only for training/validation
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def prepare_dataloader(merged_data, batch_size=32):
    # Create dataset
    dataset = MultimodalDataset(merged_data)

    # Save the dataset and batch size
    torch.save({
        "dataset": dataset,
        "batch_size": batch_size
    }, "../data/dataloader_metadata.pth")

    print("Processed data saved at ../data/dataloader_metadata.pth")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    # Paths to the CSV files
    img_path = "../data/img_train.csv"
    cgm_path = "../data/cgm_train.csv"
    viome_path = "../data/demo_viome_train.csv"
    label_path = "../data/label_train.csv"

    # Merge modalities
    print("Processing training data...")
    merged_data = merge_modalities(img_path, cgm_path, viome_path, label_path)

    # Prepare DataLoader
    dataloader = prepare_dataloader(merged_data, batch_size=32)