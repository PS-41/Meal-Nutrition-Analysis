import torch
from torch.utils.data import DataLoader
from model import MultimodalModel
from data_prep import MultimodalDataset
import pandas as pd
from preprocessing import merge_modalities

def predict_test_set(model, test_loader, device='cpu'):
    """
    Predict labels for the test set.

    Parameters:
        model (torch.nn.Module): Trained multimodal model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        list: Predictions for the test set.
    """
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # Move data to the device
            image_breakfast = batch["image_breakfast"].permute(0, 3, 1, 2).to(device)
            image_lunch = batch["image_lunch"].permute(0, 3, 1, 2).to(device)
            cgm_data = batch["cgm_data"].to(device)
            demo_data = batch["demo_viome_data"].to(device)

            # Forward pass
            outputs = model(image_breakfast, image_lunch, cgm_data, demo_data).squeeze(1)
            predictions.extend(outputs.cpu().numpy())  # Collect predictions

    return predictions

if __name__ == "__main__":
    # Paths to the CSV files for the test set
    img_test_path = "../data/img_test.csv"
    cgm_test_path = "../data/cgm_test.csv"
    viome_test_path = "../data/demo_viome_test.csv"
    label_test_path = "../data/label_test_breakfast_only.csv"  # Optional, if needed for row_id

    # Step 1: Merge modalities and prepare the test set DataLoader
    test_merged_data = merge_modalities(img_test_path, cgm_test_path, viome_test_path, label_test_path)
    test_dataset = MultimodalDataset(test_merged_data, True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Test DataLoader created successfully.")

    # Step 2: Load the trained model
    model = MultimodalModel(image_size=(64, 64, 3), cgm_size=16, demo_size=54)
    model.load_state_dict(torch.load("../results/final_multimodal_model.pth", map_location='cpu'))
    print("Trained model loaded successfully.")

    # Step 3: Predict labels for the test set
    device = 'cpu'  # Use 'cuda' if GPU is available
    predictions = predict_test_set(model, test_loader, device=device)

    # Step 4: Prepare the submission file
    row_ids = test_merged_data.index  # Use the index as row_id
    submission_df = pd.DataFrame({
        "row_id": row_ids,
        "label": predictions
    })

    # Save the submission file
    submission_file_path = "../results/test_predictions.csv"
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved at {submission_file_path}.")
