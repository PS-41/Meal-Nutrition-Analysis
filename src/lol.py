import pandas as pd
import os

# Base path to your datasets
base_path = os.path.join("..", "data")

# File paths for datasets
file_paths = {
    "Images (Train)": os.path.join(base_path, "img_train.csv"),
    "Time Series (Train)": os.path.join(base_path, "cgm_train.csv"),
    "Demographics and Microbiome (Train)": os.path.join(base_path, "demo_viome_train.csv"),
    "Labels (Train)": os.path.join(base_path, "label_train.csv")
}

# Function to display the first row of each dataset
def display_first_rows(file_paths):
    for key, path in file_paths.items():
        try:
            print(f"=== {key} Dataset ===")
            df = pd.read_csv(path)
            print("Columns:", df.columns.tolist())
            print("First Row:\n", df.iloc[0])
            print("\n")
        except Exception as e:
            print(f"Error reading {key} dataset: {e}")

# Execute the function
if __name__ == "__main__":
    display_first_rows(file_paths)
