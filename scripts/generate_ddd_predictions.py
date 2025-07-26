import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, date
from tensorflow import keras
from pathlib import Path

# Configure logging for GitHub Actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters with defaults and environment variable support."""
    month_str = os.getenv('MONTH_STR', datetime.now().strftime('%Y_%m'))
    return {
        "cwd": os.getenv('GITHUB_WORKSPACE', '.'),
        "output_dir": os.path.join(os.getenv('GITHUB_WORKSPACE', '.'), "results"),
        "month_str": month_str,
        "airflow": False
    }

def validate_data(data):
    """Validate input data format and content."""
    required_columns = ['year', 'month', 'ddd_demand', 'avg_temp_max', 'avg_temp_min', 
                       'avg_humidity', 'total_precipitation', 'total_sunshine_hours']
    
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if data.empty:
        logger.error("❌ Input data is empty")
        raise ValueError("Input data is empty")
    
    if data[['year', 'month']].isnull().any().any():
        logger.error("❌ Missing values in year or month columns")
        raise ValueError("Missing values in year or month columns")
    
    logger.info("✅ Data validation passed")
    return True

def load_data(cwd):
    """Load and prepare input data."""
    data_path = os.path.join(cwd, "data", "raw_climate_ddd_merged_data.csv")
    if not os.path.exists(data_path):
        logger.error(f"❌ Data file not found at: {data_path}")
        raise FileNotFoundError(f"raw_climate_ddd_merged_data.csv not found")

    try:
        raw_data = pd.read_csv(data_path)
        validate_data(raw_data)
        
        feature_cols = [
            'avg_temp_max', 'avg_temp_min', 'avg_humidity',
            'total_precipitation', 'total_sunshine_hours', 'ddd_demand'
        ]
        selected_data = raw_data[feature_cols].astype('float32')
        logger.info(f"✅ Loaded data with shape: {selected_data.shape}")
        return selected_data, raw_data
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise

def load_normalization_params(cwd):
    """Load normalization parameters."""
    norm_path = os.path.join(cwd, "models", "normalization_params.json")
    if not os.path.exists(norm_path):
        logger.error(f"❌ Normalization file not found at: {norm_path}")
        raise FileNotFoundError("normalization_params.json not found")

    try:
        with open(norm_path, "r") as f:
            norm_params = json.load(f)
        
        mean = np.array(norm_params["mean"], dtype=np.float32)
        std = np.array(norm_params["std"], dtype=np.float32)
        
        if len(mean) != 6 or len(std) != 6:
            logger.error("❌ Invalid normalization parameters shape")
            raise ValueError("Normalization parameters must have 6 features")
        
        std[std < 1e-10] = 1.0  # Avoid divide by zero
        logger.info("✅ Loaded normalization parameters")
        logger.info(f"Mean values: {mean}")
        logger.info(f"Std values (after clipping): {std}")
        return mean, std
    except Exception as e:
        logger.error(f"❌ Error loading normalization params: {str(e)}")
        raise

def prepare_input_window(selected_data, mean, std):
    """Prepare input window for prediction."""
    if len(selected_data) < 12:
        logger.error(f"❌ Insufficient data for input window: {len(selected_data)} rows")
        raise ValueError("Need at least 12 rows of data")
    
    input_window = selected_data.iloc[-12:].copy()
    input_raw = input_window.values  # shape: (12, 6)
    input_normalized = (input_raw - mean) / std
    input_features = input_normalized[:, :-1]  # shape: (12, 5)
    input_keras = input_features.reshape(1, 12, 5)  # shape: (1, 12, 5)
    logger.info(f"✅ Prepared input window with shape: {input_keras.shape}")
    return input_keras

def get_prediction_months(raw_data):
    """Determine target months for prediction."""
    try:
        last_year = int(raw_data["year"].iloc[-1])
        last_month = int(raw_data["month"].iloc[-1])
        prediction_months = []
        month_year_names = []
        for i in range(1, 4):
            next_month = last_month + i
            pred_year = last_year + (next_month - 1) // 12
            pred_month = ((next_month - 1) % 12) + 1
            month_name = datetime(pred_year, pred_month, 1).strftime("%B")
            prediction_months.append(f"{month_name} {pred_year}")
            month_year_names.append(f"{month_name}{pred_year}")
        logger.info(f"📆 Predicting for: {', '.join(prediction_months)}")
        return prediction_months, last_year, last_month, month_year_names
    except Exception as e:
        logger.error(f"❌ Error determining prediction months: {str(e)}")
        raise

def load_models(cwd):
    """Load prediction models."""
    model_paths = {
        'Dense': os.path.join(cwd, "models", "Dense_model.keras"),
        'GRU': os.path.join(cwd, "models", "GRU_model.keras"),
        'LSTM': os.path.join(cwd, "models", "LSTM_model.keras"),
        'transformer': os.path.join(cwd, "models", "transformer_model.keras"),
    }
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = keras.models.load_model(path)
            logger.info(f"✅ Loaded model: {name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load model {name}: {e}")
    if not models:
        logger.error("❌ No models loaded successfully")
        raise ValueError("No models loaded successfully")
    return models

def make_predictions(models, input_keras, ddd_mean, ddd_std, prediction_months):
    """Make predictions with loaded models."""
    for name, model in models.items():
        try:
            y_pred = model.predict(input_keras, verbose=0).flatten()
            y_pred_orig = y_pred * ddd_std + ddd_mean
            logger.info(f"\n📈 {name} Predictions (ddd_demand):")
            for i, month in enumerate(prediction_months):
                logger.info(f"{month}: {y_pred_orig[i]:.3f}")
        except Exception as e:
            logger.error(f"❌ Error predicting with {name}: {str(e)}")

def save_predictions(models, input_keras, ddd_mean, ddd_std, last_year, last_month, output_dir, month_year_names):
    """Save predictions to CSV."""
    prediction_records = []
    for name, model in models.items():
        try:
            y_pred = model.predict(input_keras, verbose=0).flatten()
            y_pred_orig = y_pred * ddd_std + ddd_mean
            for i, y in enumerate(y_pred_orig):
                pred_year = last_year + ((last_month + i) // 12)
                pred_month = ((last_month + i) % 12) + 1
                pred_date = date(pred_year, pred_month, 1).isoformat()
                prediction_records.append({
                    "model_name": name,
                    "date": pred_date,
                    "predicted_demand": round(float(y), 4)
                })
        except Exception as e:
            logger.error(f"❌ Error predicting with {name}: {str(e)}")

    if not prediction_records:
        logger.error("❌ No predictions generated")
        raise ValueError("No predictions generated")

    logger.info(f"Number of prediction records: {len(prediction_records)}")
    pred_df = pd.DataFrame(prediction_records)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"ddd_predictions_{'_'.join(month_year_names)}.csv"
    output_path = os.path.join(output_dir, output_filename)
    pred_df.to_csv(output_path, index=False)
    logger.info(f"💾 Saved predictions to {output_path}")

    if not os.path.exists(output_path):
        logger.error(f"❌ File not found at {output_path}")
        raise FileNotFoundError(f"Prediction file not found at {output_path}")
    
    return pred_df

def main():
    """Main function to run the prediction pipeline."""
    parser = argparse.ArgumentParser(description="Generate monthly DDD predictions")
    parser.add_argument('--month-str', type=str, default=None, 
                       help='Month string in YYYY_MM format')
    args = parser.parse_args()

    try:
        parameters = load_parameters()
        if args.month_str:
            parameters['month_str'] = args.month_str
            
        cwd = parameters["cwd"]
        logger.info(f"📁 Working directory: {cwd}")

        selected_data, raw_data = load_data(cwd)
        mean, std = load_normalization_params(cwd)
        input_keras = prepare_input_window(selected_data, mean, std)
        prediction_months, last_year, last_month, month_year_names = get_prediction_months(raw_data)
        models = load_models(cwd)
        make_predictions(models, input_keras, mean[-1], std[-1], prediction_months)
        
        output_dir = parameters["output_dir"]
        pred_df = save_predictions(models, input_keras, mean[-1], std[-1], 
                                 last_year, last_month, output_dir, month_year_names)
        
        if not parameters.get("airflow", False):
            print(pred_df.to_string())
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
