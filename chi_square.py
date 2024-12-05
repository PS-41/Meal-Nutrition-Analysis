import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def perform_chi_square(merged_data):
    """
    Perform a Chi-square test to find the correlation between numerical input features and the target label.

    Parameters:
        merged_data (pd.DataFrame): Merged dataset containing input features and labels.

    Returns:
        pd.DataFrame: DataFrame containing features and their chi-square scores with p-values.
    """
    # Exclude image data from the dataset (assuming 'Image Before Breakfast' and 'Image Before Lunch' are column names)
    exclude_columns = ["Image Before Breakfast", "Image Before Lunch"]

    # Select only the relevant numerical columns for chi-square test
    X = merged_data.drop(columns=["Lunch Calories"] + exclude_columns)
    y = merged_data["Lunch Calories"]

    # Ensure only numerical or categorical data are included
    X = X.select_dtypes(include=[np.number])

    # Scale numerical features to be in range [0, 1] as required by chi2
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform the chi-square test
    chi2_scores, p_values = chi2(X_scaled, y)

    # Create a DataFrame to store the results
    chi2_results = pd.DataFrame({
        'Feature': X.columns,
        'Chi2 Score': chi2_scores,
        'P-value': p_values
    })

    # Sort features by their chi-square scores
    chi2_results_sorted = chi2_results.sort_values(by='Chi2 Score', ascending=False)

    return chi2_results_sorted

if __name__ == "__main__":
    # Load the merged dataset
    img_path = "../data/img_train.csv"
    cgm_path = "../data/cgm_train.csv"
    viome_path = "../data/demo_viome_train.csv"
    label_path = "../data/label_train.csv"

    # Merge modalities to create a combined dataset
    from preprocessing import merge_modalities
    merged_data = merge_modalities(img_path, cgm_path, viome_path, label_path)

    # Perform the Chi-square test
    chi2_results = perform_chi_square(merged_data)

    # Print the Chi-square scores and p-values
    print("Chi-square Test Results:")
    print(chi2_results)
