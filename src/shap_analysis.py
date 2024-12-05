import torch
from data_prep import MultimodalDataset
from preprocessing import merge_modalities
from model import MultimodalModel
from model_interpretability import explain_model_with_shap, visualize_shap_values, WrappedModel

if __name__ == "__main__":
    # Load the dataset
    metadata = torch.load("../data/dataloader_metadata.pth")
    dataset = metadata["dataset"]

    # Step 3: Load the trained model
    model = MultimodalModel(
        image_size=(64, 64, 3),
        cgm_size=16,
        demo_size=41,
        dropout_rate=0.5
    )
    model.load_state_dict(torch.load("../results/multimodal_model.pth"))
    model.eval()  # Set the model to evaluation mode
    print("Trained model loaded successfully.")

    # Wrap the model for SHAP compatibility
    wrapped_model = WrappedModel(model)

    # Step 4: Perform SHAP analysis
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shap_values, inputs = explain_model_with_shap(wrapped_model, dataset, num_samples=100, device=device)

    # Step 5: Visualize SHAP results
    visualize_shap_values(shap_values, inputs)
