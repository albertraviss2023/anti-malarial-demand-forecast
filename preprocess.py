import pandas as pd
import numpy as np
import logging
from config import Config

# Setup logging
logging.basicConfig(
    filename='preprocess.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_data(input_path: str, output_path: str) -> None:
    """Preprocess raw input data and save to output path"""
    try:
        # Load raw data without headers
        df = pd.read_csv(input_path, names=Config.EXPECTED_COLS)
        logger.info(f"Loaded raw data from {input_path}")
        
        # Validate columns
        if df.shape[1] != len(Config.EXPECTED_COLS):
            raise ValueError(f"Expected {len(Config.EXPECTED_COLS)} columns, got {df.shape[1]}")
        
        # Assign dates from Jan 2017 to Mar 2025 (99 months)
        start_date = pd.to_datetime('2017-01-01')
        num_months = len(df)
        if num_months != 99:
            raise ValueError(f"Expected exactly 99 months (Jan 2017 - Mar 2025), got {num_months}")
        dates = pd.date_range(start=start_date, periods=num_months, freq='MS')
        df.index = dates
        
        # Select last 12 months (Apr 2024 to Mar 2025)
        df = df.iloc[-12:].copy()
        
        # Handle missing values
        if df.isnull().any().any():
            logger.warning("Missing values detected, filling with column means")
            df.fillna(df.mean(), inplace=True)
        
        # Ensure correct data types
        df = df.astype({col: 'float32' for col in Config.EXPECTED_COLS})
        
        # Validate data ranges
        if (df['avg_temp_max'] < -50).any() or (df['avg_temp_max'] > 50).any():
            logger.warning("Unusual temperature values detected in avg_temp_max")
        if (df['avg_humidity'] < 0).any() or (df['avg_humidity'] > 100).any():
            raise ValueError("Invalid humidity values: must be between 0 and 100")
        if (df['total_precipitation'] < 0).any() or (df['total_sunshine_hours'] < 0).any():
            raise ValueError("Negative precipitation or sunshine hours detected")
        if (df['ddd_demand'] < 0).any():
            raise ValueError("Negative demand values detected")
        
        # Save preprocessed data with date index
        df.to_csv(output_path, index=True, index_label='date')
        logger.info(f"Saved preprocessed data to {output_path}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    input_path = os.path.join(Config.DATA_DIR, "selected_data.csv")
    output_path = os.path.join(Config.DATA_DIR, "input_data.csv")
    preprocess_data(input_path, output_path)