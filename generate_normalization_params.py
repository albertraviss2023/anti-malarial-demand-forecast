import pandas as pd
import numpy as np
import json
import os

# Define paths and columns
DATA_DIR = 'data'
MODELS_DIR = 'models'
FEATURE_COLS = [
    'avg_temp_max', 'avg_temp_min', 'avg_humidity',
    'total_precipitation', 'total_sunshine_hours', 'ddd_demand'
]

def generate_normalization_params(input_path: str, output_path: str) -> None:
    """Generate normalization parameters (mean, std) from training data and save to JSON."""
    try:
        # Load data with headers
        df = pd.read_csv(input_path, header=0)
        print("Column order in selected_data:")
        print(df.columns.tolist())
        
        # Verify columns match expected
        if not all(col in df.columns for col in FEATURE_COLS):
            raise ValueError(f"Expected columns {FEATURE_COLS}, got {df.columns.tolist()}")
        
        # Convert to float32
        data_values = df[FEATURE_COLS].values.astype('float32')
        
        # Calculate split sizes
        num_samples = len(data_values)
        num_train = int(0.60 * num_samples)
        
        print(f"Total samples: {num_samples}")
        print(f"Train samples: {num_train}")
        
        # Compute mean and std on training data only
        mean = data_values[:num_train].mean(axis=0)
        std = data_values[:num_train].std(axis=0)
        std[std < 1e-10] = 1.0  # Prevent division by zero
        
        # Save to JSON
        params = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"Saved normalization parameters to {output_path}")
        print("Mean:", params['mean'])
        print("Std:", params['std'])
        
    except Exception as e:
        print(f"Error generating normalization parameters: {str(e)}")
        raise

if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, 'selected_data.csv')
    output_path = os.path.join(MODELS_DIR, 'normalization_params.json')
    generate_normalization_params(input_path, output_path)