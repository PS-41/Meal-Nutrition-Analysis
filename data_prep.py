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
        Initializing the MultimodalDataset to process all input sets.
        
        Parameters:
            merged_data (pd.DataFrame): Merged and preprocessed dataset.
            is_test (bool): Indicates whether the dataset is a test set because it would have no labels.
        """
        self.images_breakfast = preprocess_images(merged_data["Image Before Breakfast"])
        self.images_lunch = preprocess_images(merged_data["Image Before Lunch"])
        self.cgm_data = preprocess_cgm(merged_data["CGM Data"])
        self.demo_viome_data = preprocess_demo_viome(merged_data)
        self.is_test = is_test 

        if not is_test:
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
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def prepare_dataloader(merged_data, batch_size=32):
    dataset = MultimodalDataset(merged_data)

    # Saving the dataset and batch size only instead of the entire DataLoader
    torch.save({
        "dataset": dataset,  
        "batch_size": batch_size  
    }, "../data/dataloader_metadata.pth")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    img_path = "../data/img_train.csv"
    cgm_path = "../data/cgm_train.csv"
    viome_path = "../data/demo_viome_train.csv"
    label_path = "../data/label_train.csv"

    # Merging the modalities
    merged_data = merge_modalities(img_path, cgm_path, viome_path, label_path)

    # Preparing the DataLoader
    dataloader = prepare_dataloader(merged_data, batch_size=32)

    # Testing the DataLoader
    total_rows = 0
    for batch in dataloader:
        total_rows += len(batch["label"])  

    print(f"Total number of rows in the DataLoader: {total_rows}")

    sample_batch = next(iter(dataloader)) 
    print(f"Column names in the DataLoader: {sample_batch.keys()}")

    batch = next(iter(dataloader)) 

    # Verifying the Image Data
    print(f"Image Before Breakfast Shape: {batch['image_breakfast'].shape}")  
    print(f"Image Before Lunch Shape: {batch['image_lunch'].shape}")  

    # Verifing the CGM Data
    print(f"CGM Data Shape: {batch['cgm_data'].shape}")  

    # Verifying the Demographic and Microbiome Data
    demo_viome_data_shape = batch["demo_viome_data"].shape
    print(f"Demo and Viome Data Shape: {demo_viome_data_shape}")  
    print(f"Actual Demo and Viome Columns: {demo_viome_data_shape[1]}")  

    # Verifying the Target Variable
    print(f"Labels Shape: {batch['label'].shape}")
