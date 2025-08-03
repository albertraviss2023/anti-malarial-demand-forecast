#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Malaria Case Prediction Script

This script loads historical malaria and climate data, automatically determines
the latest available data point, and generates predictions for the next three
months. It applies five different trained models (Dense, LSTM, GRU, Transformer, XGBoost)
to generate forecasts, and saves the predictions to a single CSV file in long format.

The filename will dynamically include the predicted months (e.g., predictions_May_2025_June_2025_July_2025.csv).

It is designed to be run as a single, automated process, ideal for production
environments where input data is updated regularly.

Execution:
    python generate_predictions.py
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from datetime import datetime
import sys # Import sys for StreamHandler in logging setup

# Import from Keras directly for custom layer registration
from keras.layers import Layer
from keras.saving import register_keras_serializable


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- File and Directory Paths ---
# The script assumes a directory structure like:
# /project/
#   |- scripts/
#   |   |- generate_predictions.py  (this script)
#   |- data/
#   |   |- malaria_historical.csv
#   |- malaria_models/
#   |   |- dense.keras, LSTM.keras, etc.
#   |- results/
#       |- (output files will be saved here)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "malaria_models"
RESULT_DIR = BASE_DIR / "results"

# --- Input/Output Files ---
# This is the input data file name.
DATA_FILENAME = "malaria_historical.csv" 
# The output filename will be generated dynamically in the main function.

# --- Logging Setup ---
LOG_FILE = RESULT_DIR / "prediction.log"
RESULT_DIR.mkdir(exist_ok=True) # Ensure result directory exists

# Configure logging:
# - Set encoding for the FileHandler to UTF-8 to correctly write any Unicode characters.
# - Use basic StreamHandler for console output (emojis are removed from messages below).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), # Ensure log file uses UTF-8
        logging.StreamHandler(sys.stdout) # Explicitly use sys.stdout for the console stream
    ]
)

# =============================================================================
# 2. KERAS CUSTOM LAYER DEFINITION
# =============================================================================

@register_keras_serializable()
class CastAndClipLayer(Layer):
    """Custom layer required for loading the Keras models."""
    def __init__(self, num_districts=118, **kwargs):
        super().__init__(**kwargs)
        self.num_districts = num_districts

    def call(self, inputs):
        clipped = tf.clip_by_value(inputs, 0, self.num_districts - 1)
        return tf.cast(clipped, tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_districts": self.num_districts,
        })
        return config

# =============================================================================
# 3. DATA LOADING AND PRE-PROCESSING FUNCTIONS
# =============================================================================

def load_models(model_dir: Path, custom_objects: dict) -> dict:
    """Loads all Keras and XGBoost models from the specified directory."""
    models = {}
    model_paths = {
        'Dense': model_dir / "dense.keras",
        'LSTM': model_dir / "LSTM.keras",
        'GRU': model_dir / "GRU.keras",
        'Transformer': model_dir / "transformer.keras",
        'XGBoost': model_dir / "XGBoost.json",
    }

    for name, path in model_paths.items():
        if not path.exists():
            logging.warning(f"Model file not found for '{name}' at: {path}")
            continue
        try:
            if name == 'XGBoost':
                booster = xgb.Booster()
                booster.load_model(str(path))
                models[name] = booster
                logging.info(f"Loaded XGBoost model '{name}'")
            else:
                models[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
                logging.info(f"Loaded Keras model '{name}'")
        except Exception as e:
            logging.error(f"Failed to load model '{name}' from {path}: {e}")
    return models

def calculate_normalization_stats(df: pd.DataFrame) -> dict:
    """Calculates mean and std for each district based on the first 60% of its data."""
    stats_per_district = {}
    feature_cols = [
        'avg_temp_max', 'avg_temp_min', 'avg_humidity',
        'sum_precipitation', 'sum_sunshine_hours', 'mal_cases'
    ]
    for district, df_district in df.groupby('district'):
        df_district = df_district.sort_values(['year', 'month']).reset_index(drop=True)
        data_values = df_district[feature_cols].values.astype('float32')
        
        num_train = int(0.60 * len(data_values))
        mean = data_values[:num_train].mean(axis=0)
        std = data_values[:num_train].std(axis=0)
        std[std < 1e-10] = 1.0  # Avoid division by zero
        
        stats_per_district[district] = {'mean': mean, 'std': std}
    logging.info(f"Calculated normalization stats for {len(stats_per_district)} districts.")
    return stats_per_district

# =============================================================================
# 4. PREDICTION PIPELINE FUNCTIONS
# =============================================================================

def prepare_input_data(df, stats, district_map, input_window_start_date_str, input_window_end_date_str):
    """
    Prepares and normalizes the input data for all districts for a given time window.
    The input window is expected to be 6 months.
    """
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    start = pd.to_datetime(input_window_start_date_str + '-01')
    end = pd.to_datetime(input_window_end_date_str + '-01')
    
    input_df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    
    input_data, valid_districts = [], []
    feature_cols = ['avg_temp_max', 'avg_temp_min', 'avg_humidity', 'sum_precipitation', 'sum_sunshine_hours']
    
    for district, district_id in district_map.items():
        district_data = input_df[input_df['district'] == district].sort_values('date')
        if len(district_data) != 6: # Ensure we have exactly 6 months of data for the prediction window
            logging.warning(f"Skipping district {district}: insufficient data for prediction input window "
                            f"({len(district_data)} months found, 6 required).")
            continue
            
        district_features = district_data[feature_cols].to_numpy()
        mean, std = stats[district]['mean'][:-1], stats[district]['std'][:-1]
        normalized_features = (district_features - mean) / std
        
        id_feature = np.full((6, 1), district_id)
        district_input = np.concatenate([normalized_features, id_feature], axis=-1)
        
        input_data.append(district_input)
        valid_districts.append(district)
    
    if not valid_districts:
        raise ValueError("No districts have complete data for the specified input window. Aborting prediction.")
        
    return np.array(input_data), valid_districts

def prepare_xgboost_data(batch_features, num_districts):
    """Converts sequential data to a flat format for XGBoost."""
    regular_features = batch_features[:, :, :-1]
    district_ids = batch_features[:, :, -1][:, -1].astype(np.int32)
    
    one_hot_encoder = tf.keras.layers.CategoryEncoding(num_tokens=num_districts, output_mode="one_hot")
    district_ids_encoded = one_hot_encoder(district_ids).numpy()
    
    regular_features_flat = regular_features.reshape(batch_features.shape[0], -1)
    return np.concatenate([regular_features_flat, district_ids_encoded], axis=-1)

def make_predictions(models_dict, input_data, valid_districts, stats, num_districts):
    """Generates and denormalizes predictions for all models."""
    predictions = {}
    for name, model in models_dict.items():
        if name == 'XGBoost':
            X_flat = prepare_xgboost_data(input_data, num_districts)
            dmatrix_pred = xgb.DMatrix(X_flat)
            y_pred = model.predict(dmatrix_pred)
        else:
            y_pred = model.predict(input_data, verbose=0)
            
        denorm_preds = np.zeros_like(y_pred)
        for i, district in enumerate(valid_districts):
            mean, std = stats[district]['mean'][-1], stats[district]['std'][-1]
            denorm_preds[i] = y_pred[i] * std + mean
        
        predictions[name] = denorm_preds
        logging.info(f"Generated predictions for model: {name}")
    return predictions

# =============================================================================
# 5. EXPORT FUNCTIONS
# =============================================================================

def export_long_format(predictions, districts, output_path, forecast_months):
    """
    Exports predictions to a single CSV in long format (district, model, month, predicted_mal_cases).
    This is the primary output format.
    """
    output_data = []
    
    for i, district in enumerate(districts):
        for model_name, preds_array in predictions.items():
            for month_idx, month_str in enumerate(forecast_months):
                output_data.append({
                    'district': district,
                    'model': model_name,
                    'month': month_str,
                    'predicted_mal_cases': max(0, preds_array[i, month_idx]) # Ensure non-negative predictions
                })
    pd.DataFrame(output_data).to_csv(output_path, index=False, float_format='%.2f')
    logging.info(f"All predictions saved to single long-format file: {output_path}")

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the entire prediction pipeline."""
    logging.info("Starting malaria prediction pipeline...")
    
    # --- Step 1: Load Data and Models ---
    try:
        data_path = DATA_DIR / DATA_FILENAME # DATA_FILENAME is now correctly defined
        raw_data = pd.read_csv(data_path)
        logging.info(f"Successfully loaded data from {data_path}")
        
        models = load_models(MODEL_DIR, {'CastAndClipLayer': CastAndClipLayer})
        if not models:
            raise RuntimeError("No models were loaded. Aborting.")
            
    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"FATAL: Failed during initial setup. {e}")
        return

    # --- Step 2: Determine Prediction Window Dynamically ---
    # Convert 'year' and 'month' to datetime objects to find the latest date
    raw_data['date'] = pd.to_datetime(raw_data[['year', 'month']].assign(day=1))
    latest_date_in_data = raw_data['date'].max()
    logging.info(f"Latest historical data available up to: {latest_date_in_data.strftime('%Y-%m')}")

    # The 6-month input window ends with the latest available data month
    input_window_end_date_str = latest_date_in_data.strftime('%Y-%m')
    # The input window starts 5 months prior to the end date (to get a 6-month window)
    input_window_start_date_str = (latest_date_in_data - pd.DateOffset(months=5)).strftime('%Y-%m')
    logging.info(f"Using input data window from {input_window_start_date_str} to {input_window_end_date_str}.")

    # Determine the 3 forecast months
    forecast_months_list = []
    for i in range(1, 4): # For 3 months (1, 2, 3 months after latest_date_in_data)
        forecast_date = latest_date_in_data + pd.DateOffset(months=i)
        forecast_months_list.append(forecast_date.strftime('%B_%Y')) # e.g., 'May_2025'
    logging.info(f"Forecasting for months: {', '.join(forecast_months_list)}")

    # --- Step 3: Pre-process and Prepare Inputs ---
    if 'Kamuli' in raw_data['district'].unique():
        raw_data = raw_data[raw_data['district'] != 'Kamuli'].copy()
        logging.info("Removed 'Kamuli' district from the dataset (if present).")

    unique_districts = sorted(raw_data['district'].unique())
    district_to_id = {name: i for i, name in enumerate(unique_districts)}
    num_districts = len(unique_districts)

    stats = calculate_normalization_stats(raw_data)
    
    try:
        input_data, valid_districts = prepare_input_data(
            raw_data, stats, district_to_id, input_window_start_date_str, input_window_end_date_str
        )
        logging.info(f"Prepared input data for {len(valid_districts)} districts for prediction.")
    except ValueError as e:
        logging.error(f"FATAL: Could not prepare input data. {e}")
        return
        
    # --- Step 4: Generate Predictions ---
    all_predictions = make_predictions(models, input_data, valid_districts, stats, num_districts)
    
    # --- Step 5: Export Results ---
    # Construct the dynamic filename using the forecast months
    predicted_months_filename_part = "_".join(forecast_months_list)
    dynamic_output_filename = f"predictions_{predicted_months_filename_part}.csv"
    
    # Export the long format with the dynamic filename
    export_long_format(all_predictions, valid_districts, RESULT_DIR / dynamic_output_filename, forecast_months_list)

    logging.info("Pipeline finished successfully!")


if __name__ == '__main__':
    main()
    