import pandas as pd
import ast
from ast import literal_eval

import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2

def merge_modalities(img_path, cgm_path, viome_path, label_path):
    """
    Merges data from all modalities into a single dataframe based on Subject ID and Day.

    Parameters:
        img_path (str): Path to the img_train.csv file.
        cgm_path (str): Path to the cgm_train.csv file.
        viome_path (str): Path to the demo_viome_train.csv file.
        label_path (str): Path to the label_train.csv file.

    Returns:
        pd.DataFrame: A merged dataframe containing all modalities.
    """
    # Load the CSV files
    img_data = pd.read_csv(img_path)
    cgm_data = pd.read_csv(cgm_path)
    viome_data = pd.read_csv(viome_path)
    label_data = pd.read_csv(label_path)

    # Convert CGM Data to lists of tuples
    cgm_data['CGM Data'] = cgm_data['CGM Data'].apply(literal_eval)

    # Merge all dataframes on Subject ID and Day
    merged = (
        img_data
        .merge(cgm_data, on=["Subject ID", "Day"], how="inner")
        .merge(viome_data, on=["Subject ID"], how="inner")
        .merge(label_data, on=["Subject ID", "Day"], how="inner")
    )

    return merged

def preprocess_images(image_column, image_size=(64, 64)):
    """
    Preprocesses an image column by resizing and normalizing images.
    Handles missing images by replacing them with a placeholder.
    """
    processed_images = []
    placeholder_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)

    for image in image_column:
        if not image or image == "[]":
            # Handle missing images by adding a placeholder
            processed_images.append(placeholder_image)
        else:
            try:
                # Convert string representation of the image to a Python list
                if isinstance(image, str):
                    image = ast.literal_eval(image)
                
                # Convert the image to a numpy array if not already
                image_array = np.array(image, dtype=np.float32)
                
                # Normalize pixel values to [0, 1]
                image_array /= 255.0
                
                # Resize the image to the desired size (if necessary)
                if image_array.shape[:2] != image_size:
                    from skimage.transform import resize
                    image_array = resize(image_array, image_size, anti_aliasing=True)
                
                processed_images.append(image_array)
            except Exception as e:
                print(f"Error processing image: {e}")
                # Append placeholder if an error occurs
                processed_images.append(placeholder_image)
    
    return np.array(processed_images, dtype=np.float32)

def preprocess_cgm(cgm_column, resample_freq='30T', max_length=16):
    """
    Preprocess CGM data for time-series modeling.
    
    Parameters:
        cgm_column (pd.Series): Column containing CGM data as lists of tuples.
        resample_freq (str): Resampling frequency (e.g., '30T' for 30 minutes).
        max_length (int): Maximum length of the time series (padding or truncating).
        
    Returns:
        np.ndarray: Processed CGM data with fixed-length sequences.
    """
    processed_cgm = []
    for row in cgm_column:
        if not row or row == []:  # Handle missing or empty CGM data
            # Placeholder for missing CGM data
            placeholder = [0] * max_length
            processed_cgm.append(placeholder)
            continue
        
        # Confirm the data is already in list format
        if isinstance(row, str):
            # Parse string representation into list of tuples
            cgm_data = literal_eval(row)
        else:
            cgm_data = row  # Use the row as-is if it's already a list of tuples
        
        # Extract time and glucose values
        times, glucose_values = zip(*cgm_data)
        times = pd.to_datetime(times)
        glucose_values = np.array(glucose_values, dtype=np.float32)
        
        # Create a DataFrame for resampling
        df = pd.DataFrame({'Glucose': glucose_values}, index=times)
        
        # Resample to the desired frequency and interpolate missing values
        df = df.resample(resample_freq).mean().interpolate()
        
        # Truncate or pad to ensure fixed length
        truncated_or_padded = df['Glucose'].iloc[:max_length].to_list()
        if len(truncated_or_padded) < max_length:
            # Pad with 0s if shorter than max_length
            truncated_or_padded.extend([0] * (max_length - len(truncated_or_padded)))
        
        # Normalize glucose values to [0, 1]
        normalized = np.array(truncated_or_padded) / 300.0  # Assuming max glucose = 300 mg/dL
        processed_cgm.append(normalized)
    
    return np.array(processed_cgm)

def preprocess_demo_viome(demo_viome_data):
    """
    Preprocess demographic and microbiome data.

    Parameters:
        demo_viome_data (pd.DataFrame): The demographic and microbiome data.

    Returns:
        pd.DataFrame: Processed demographic and microbiome features.
    """
    # Define columns
    categorical_cols = ['Gender', 'Race', 'Diabetes Status']
    numerical_cols = [
        'Age', 'Weight', 'Height', 'BMI', 'A1C', 
        'Baseline Fasting Glucose', 'Insulin', 'Triglycerides', 
        'Cholesterol', 'HDL', 'Non-HDL', 'LDL', 'VLDL', 
        'CHO/HDL Ratio', 'HOMA-IR',
        'Breakfast Time', 'Lunch Time', 
        'Breakfast Calories', 'Breakfast Carbs', 
        'Breakfast Fat', 'Breakfast Protein'
    ]
    microbiome_col = 'Viome'  # Explicitly handle the Viome column

    # Function to calculate the mean time (in minutes) for the same subject across all days
    def fill_time_with_subject_mean(row, time_col):
        if pd.isna(row[time_col]):
            # Calculate the mean time for the same subject across all days
            subject_mean_time = demo_viome_data[
                (demo_viome_data['Subject ID'] == row['Subject ID']) & demo_viome_data[time_col].notna()
            ][time_col].mean()
            return subject_mean_time if not pd.isna(subject_mean_time) else 0  # Default to 0 if no mean available
        return row[time_col]

    # Handle time columns
    for time_col in ['Breakfast Time', 'Lunch Time']:
        # Convert time to minutes past midnight, handling invalid entries
        demo_viome_data[time_col] = pd.to_datetime(
            demo_viome_data[time_col], errors='coerce', format='%Y-%m-%d %H:%M:%S'
        ).dt.hour * 60 + pd.to_datetime(demo_viome_data[time_col], errors='coerce').dt.minute

        # Fill missing or invalid times with subject-specific mean
        demo_viome_data[time_col] = demo_viome_data.apply(
            lambda row: fill_time_with_subject_mean(row, time_col), axis=1
        )

    # Expand the Viome column into separate numerical features
    viome_features = demo_viome_data[microbiome_col].str.split(',', expand=True)
    viome_features.columns = [f'Viome_{i+1}' for i in range(viome_features.shape[1])]
    viome_features = viome_features.astype(float)  # Ensure numerical type

    # Combine Viome features with the main data
    demo_viome_data_expanded = pd.concat([demo_viome_data, viome_features], axis=1)
    demo_viome_data_expanded.drop(columns=[microbiome_col, 'Subject ID'], inplace=True)

    # Perform feature selection
    target_column = 'Lunch Calories'
    selected_categorical = select_categorical_features(demo_viome_data_expanded, target_column, categorical_cols)
    selected_numerical = select_numerical_features(demo_viome_data_expanded, target_column, numerical_cols)
    selected_viome = select_numerical_features(demo_viome_data_expanded, target_column, viome_features.columns.tolist())

    unselected_numerical = [col for col in numerical_cols if col not in selected_numerical]
    unselected_viome = [col for col in viome_features.columns if col not in selected_viome]
    
    # Apply PCA to unselected features
    pca_df = apply_pca(demo_viome_data_expanded, unselected_numerical, unselected_viome, explained_variance=0.95, save_plots=True, plot_dir="../results/")

    # Keep the categorical features since gender, race and diabetes information are relevant to lunch calories from domain knowledge
    if not selected_categorical:
        selected_categorical = categorical_cols

    # Preprocessing pipelines
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, selected_numerical + selected_viome),
            ('cat', categorical_pipeline, selected_categorical)
        ]
    )
    
    # Apply transformations
    filtered_data = demo_viome_data_expanded[selected_numerical + selected_viome + selected_categorical]
    processed_features = preprocessor.fit_transform(filtered_data)

    # Convert to DataFrame
    processed_feature_names = (
        selected_numerical + selected_viome +
        list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(selected_categorical))
    )

    processed_df = pd.DataFrame(processed_features, columns=processed_feature_names)

    # Combine selected features with PCA-transformed features
    final_features_df = pd.concat([processed_df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    
    return final_features_df

def select_categorical_features(demo_viome_data, target_column, categorical_cols, p_threshold=0.05):
    """
    Select relevant categorical features using the chi-square test or mutual information.

    Parameters:
        demo_viome_data (pd.DataFrame): The dataset containing categorical features.
        target_column (str): The name of the target column.
        categorical_cols (list): List of categorical columns to evaluate.
        p_threshold (float): Significance level for chi-square feature selection.

    Returns:
        list: Selected categorical feature names.
    """
    from sklearn.feature_selection import chi2, mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    # Encode categorical features
    encoded_data = demo_viome_data[categorical_cols].apply(LabelEncoder().fit_transform)

    # Convert the target to categorical if necessary
    target = demo_viome_data[target_column]
    if not pd.api.types.is_categorical_dtype(target):
        # Create bins for the target if numerical
        target = pd.cut(target, bins=3, labels=[0, 1, 2])  # 3 bins (adjust as needed)

    # Perform chi-square test
    chi_scores, p_values = chi2(encoded_data, target)

    # Select features with significant p-values
    selected_features = [
        feature for feature, p_val in zip(categorical_cols, p_values) if p_val < p_threshold
    ]
    print("p = ", p_values)

    # Fallback to mutual information if no features selected
    if not selected_features:
        print("Fallback to mutual information for categorical features.")
        mi_scores = mutual_info_classif(encoded_data, target)
        mi_df = pd.DataFrame({
            "Feature": categorical_cols,
            "Mutual_Info_Score": mi_scores
        }).sort_values(by="Mutual_Info_Score", ascending=False)
        selected_features = mi_df[mi_df["Mutual_Info_Score"] > 0.01]["Feature"].tolist()
        print(mi_df)
    print("\n\n selected features num = \n\n", selected_features)
    

    return selected_features


def select_numerical_features(demo_viome_data, target_column, numerical_cols, corr_threshold=0.1):
    """
    Select relevant numerical features based on correlation with the target variable.

    Parameters:
        demo_viome_data (pd.DataFrame): The dataset containing numerical features.
        target_column (str): The name of the target column.
        numerical_cols (list): List of numerical columns to evaluate.
        corr_threshold (float): Absolute correlation threshold for feature selection.

    Returns:
        list: Selected numerical feature names.
    """
    # Compute correlation
    correlations = demo_viome_data[numerical_cols].corrwith(demo_viome_data[target_column])

    # Filter features based on correlation threshold
    selected_features = correlations[abs(correlations) > corr_threshold].index.tolist()

    print(correlations)

    print("\n\n selected features num = \n\n", selected_features)
    return selected_features

def apply_pca(demo_viome_data, unselected_numerical, unselected_viome, explained_variance=0.95, save_plots=False, plot_dir="plots"):
    """
    Apply PCA to unselected numerical and Viome features and plot explained variance.

    Parameters:
        demo_viome_data (pd.DataFrame): The demographic and microbiome data.
        unselected_numerical (list): List of unselected numerical columns.
        unselected_viome (list): List of unselected Viome columns.
        explained_variance (float): Desired cumulative explained variance for PCA.
        save_plots (bool): Whether to save plots to a directory.
        plot_dir (str): Directory to save plots.

    Returns:
        pd.DataFrame: PCA-transformed features as a DataFrame.
    """
    import os

    # Combine unselected numerical and Viome features
    pca_features = unselected_numerical + unselected_viome

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(demo_viome_data[pca_features])

    # Apply PCA to determine the number of components for the desired explained variance
    pca_temp = PCA()
    pca_temp.fit(scaled_data)
    cumulative_variance = pca_temp.explained_variance_ratio_.cumsum()
    n_components = next(i for i, ratio in enumerate(cumulative_variance) if ratio >= explained_variance) + 1

    # Plot explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca_temp.explained_variance_ratio_) + 1), pca_temp.explained_variance_ratio_, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid()
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "explained_variance_ratio.png"))
    plt.show()

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=explained_variance, color='r', linestyle='--', label=f'{explained_variance * 100:.0f}% Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.legend()
    plt.grid()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, "cumulative_explained_variance.png"))
    plt.show()

    print("\n\nn componenet = ", n_components)
    # Apply PCA with the determined number of components
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    # Create DataFrame for PCA-transformed features
    pca_columns = [f"PCA_{i+1}" for i in range(pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=demo_viome_data.index)

    return pca_df