import os

class Config:
    """Configuration settings for the application"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    
    MODEL_FILES = {
        'Dense': 'Dense_model.keras',
        'GRU': 'GRU_model.keras',
        'LSTM': 'LSTM_model.keras'
    }
    
    EXPECTED_COLS = [
        'avg_temp_max',
        'avg_temp_min',
        'avg_humidity',
        'total_precipitation',
        'total_sunshine_hours',
        'ddd_demand'
    ]