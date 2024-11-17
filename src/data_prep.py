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
        self.demo_viome_data = preprocess_demo_viome(merged_data)
        self.is_test = is_test  # Whether this is a test set

        if not is_test:
            # Target variable: Lunch Calories (only for training/validation)
            self.labels = merged_data["Lunch Calories"].values
        else:
            self.labels = None  # No labels for the test set

    def __len__(self):
        return len(self.cgm_data)

    def __getitem__(self, idx):
        demo_viome_row = self.demo_viome_data.iloc[idx]  # Use .iloc for row indexing
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

    # Save the dataset and batch size instead of the entire DataLoader
    torch.save({
        "dataset": dataset,  # Save the dataset object
        "batch_size": batch_size  # Save the batch size for reconstruction
    }, "../data/dataloader_metadata.pth")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    # Paths to the CSV files
    img_path = "../data/img_train.csv"
    cgm_path = "../data/cgm_train.csv"
    viome_path = "../data/demo_viome_train.csv"
    label_path = "../data/label_train.csv"

    # Step 1: Merge modalities
    merged_data = merge_modalities(img_path, cgm_path, viome_path, label_path)

    # Step 2: Prepare DataLoader
    dataloader = prepare_dataloader(merged_data, batch_size=32)

    # Step 3: Test DataLoader
    # Verify the total number of rows in the DataLoader
    total_rows = 0
    for batch in dataloader:
        total_rows += len(batch["label"])  # Summing up the batch sizes

    print(f"Total number of rows in the DataLoader: {total_rows}")

    # Verify the column names in the DataLoader
    sample_batch = next(iter(dataloader))  # Get the first batch
    print(f"Column names in the DataLoader: {sample_batch.keys()}")


    # Verify DataLoader Keys
    batch = next(iter(dataloader))  # Extract the first batch

    # Step 1: Verify Image Data
    print(f"Image Before Breakfast Shape: {batch['image_breakfast'].shape}")  # Should be (batch_size, 64, 64, 3)
    print(f"Image Before Lunch Shape: {batch['image_lunch'].shape}")  # Should be (batch_size, 64, 64, 3)

    # Step 2: Verify CGM Data
    print(f"CGM Data Shape: {batch['cgm_data'].shape}")  # Should be (batch_size, 16)

    # Step 3: Verify Demographic and Microbiome Data
    demo_viome_data_shape = batch["demo_viome_data"].shape
    print(f"Demo and Viome Data Shape: {demo_viome_data_shape}")  # Should include all demographic + microbiome columns
    # expected_demo_viome_cols = [
    #     "Age", "Weight", "Height", "BMI", "A1C", "Baseline Fasting Glucose", "Insulin", 
    #     "Triglycerides", "Cholesterol", "HDL", "Non-HDL", "LDL", "VLDL", "CHO/HDL Ratio", 
    #     "HOMA-IR", "Viome_1", "Viome_2", ..., "Viome_N", "Gender_0", "Gender_1", ...
    #     "Race_0", ..., "Race_M", "Diabetes Status_0", ..., "Diabetes Status_P"
    # ]
    # print(f"Expected Demo and Viome Columns: {len(expected_demo_viome_cols)}")
    print(f"Actual Demo and Viome Columns: {demo_viome_data_shape[1]}")  # Compare count

    # Step 4: Verify Target Variable
    print(f"Labels Shape: {batch['label'].shape}")  # Should be (batch_size,)