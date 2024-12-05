import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
import torch.nn as nn

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, *inputs):
        # SHAP passes a single input tensor or a list of tensors.
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]
        # Unpack the tuple of inputs and forward them to the original model
        image_breakfast, image_lunch, cgm_data, demo_data = inputs
        return self.model(image_breakfast, image_lunch, cgm_data, demo_data)



def explain_model_with_shap(model, dataset, num_samples=100, device='cpu'):
    """
    Use SHAP to explain predictions of a multimodal model.

    Parameters:
        model (torch.nn.Module): Trained multimodal model.
        dataset (Dataset): The dataset to explain.
        num_samples (int): Number of samples to explain.
        device (str): 'cpu' or 'cuda'.

    Returns:
        shap_values: SHAP values for the dataset.
    """
    model.to(device)
    model.eval()

    # Create a subset of the dataset for SHAP
    subset_indices = random.sample(range(len(dataset)), num_samples)
    subset = [dataset[i] for i in subset_indices]

    # Extract input tensors and ensure correct shape
    image_breakfast = torch.stack([s["image_breakfast"].permute(2, 0, 1) for s in subset]).to(device)
    image_lunch = torch.stack([s["image_lunch"].permute(2, 0, 1) for s in subset]).to(device)
    cgm_data = torch.stack([s["cgm_data"] for s in subset]).to(device)
    demo_data = torch.stack([s["demo_viome_data"] for s in subset]).to(device)

    # Combine inputs into a list for SHAP
    inputs = [image_breakfast, image_lunch, cgm_data, demo_data]

    # Wrap the model
    wrapped_model = WrappedModel(model)

    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(wrapped_model, inputs)

    # Compute SHAP values
    shap_values = explainer.shap_values(inputs)
    return shap_values, inputs


def visualize_shap_values(shap_values, inputs):
    """
    Visualize SHAP values using summary and force plots.

    Parameters:
        shap_values: SHAP values from DeepExplainer.
        inputs: Model inputs for SHAP analysis.
    """
    shap.summary_plot(shap_values, inputs, show=False)
    plt.savefig("../results/shap_summary.png")
    plt.show()

    for i in range(5):  # Visualize force plots for first 5 samples
        shap.force_plot(shap_values[i], inputs[i])

# Permutation Feature Importance
def permutation_feature_importance(model, dataset, feature_set, num_samples=100, device='cpu'):
    """
    Calculate permutation-based feature importance.

    Parameters:
        model (torch.nn.Module): Trained model.
        dataset (Dataset): Full dataset.
        feature_set (list): List of features/modalities to analyze.
        num_samples (int): Number of samples to evaluate.
        device (str): 'cpu' or 'cuda'.

    Returns:
        dict: Feature importance scores.
    """
    model.to(device)
    model.eval()

    # Randomly select samples
    subset_indices = torch.randperm(len(dataset))[:num_samples]
    subset = [dataset[i] for i in subset_indices]
    labels = torch.stack([item["label"] for item in subset]).to(device)

    # Baseline predictions
    predictions = model(subset).detach().cpu().numpy()
    baseline_rmsre = np.sqrt(mean_squared_error(labels.cpu().numpy(), predictions))

    importance_scores = {}
    for feature in feature_set:
        permuted_data = subset.copy()

        # Permute the feature
        for sample in permuted_data:
            np.random.shuffle(sample[feature])

        permuted_predictions = model(permuted_data).detach().cpu().numpy()
        permuted_rmsre = np.sqrt(mean_squared_error(labels.cpu().numpy(), permuted_predictions))

        importance_scores[feature] = permuted_rmsre - baseline_rmsre

    return importance_scores

def visualize_feature_importance(importance_scores):
    """
    Visualize feature importance as a bar plot.

    Parameters:
        importance_scores (dict): Feature importance scores.
    """
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())

    plt.figure(figsize=(10, 6))
    plt.barh(features, scores, color="skyblue")
    plt.xlabel("Change in RMSRE")
    plt.ylabel("Features")
    plt.title("Feature Importance (Permutation-Based)")
    plt.grid(True)
    plt.savefig("../results/feature_importance.png")
    plt.show()
