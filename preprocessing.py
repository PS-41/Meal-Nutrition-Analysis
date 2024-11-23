import pandas as pd
import ast
from ast import literal_eval

import numpy as np
from torchvision import transforms
from PIL import Image

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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
        'Breakfast Time', 'Lunch Time'  # Include times
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
            ('num', numerical_pipeline, numerical_cols + viome_features.columns.tolist()),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )
    
    # Apply transformations
    processed_features = preprocessor.fit_transform(demo_viome_data_expanded)
    
    # Convert to DataFrame
    processed_feature_names = (
        numerical_cols + viome_features.columns.tolist() +
        list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
    )
    processed_df = pd.DataFrame(processed_features, columns=processed_feature_names)
    
    return processed_df