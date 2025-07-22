import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, PlainTextResponse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Anti-Malarial Demand Forecasting API",
    description="API for the Anti-Malarial Demand Forecasting Dashboard",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Expected files
REQUIRED_FILES = [
    'raw_climate_ddd_merged_data.csv',
    'test_metrics.csv',
    'predictions.csv'
]

@app.on_event("startup")
async def startup_event():
    """Validate required files and directories"""
    try:
        # Create directories if they don't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATIC_DIR, exist_ok=True)
        
        # Validate required files
        for filename in REQUIRED_FILES:
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required file not found: {filepath}")
            
            # Test reading each file
            try:
                pd.read_csv(filepath)
            except Exception as e:
                raise ValueError(f"Failed to read {filename}: {str(e)}")
        
        logger.info("Startup validation completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to dashboard"""
    return RedirectResponse(url="/static/index.html")

@app.get("/api/input-data", response_class=PlainTextResponse)
async def get_input_data():
    """Get the most recent 12 months of input data as CSV"""
    try:
        filepath = os.path.join(DATA_DIR, 'raw_climate_ddd_merged_data.csv')
        df = pd.read_csv(filepath)
        
        # Validate columns
        expected_cols = ['year', 'month', 'ddd_demand', 'avg_temp_max', 
                        'avg_temp_min', 'avg_humidity', 'total_precipitation', 
                        'total_sunshine_hours']
        
        if not all(col in df.columns for col in expected_cols):
            raise ValueError("Input data missing required columns")
        
        # Convert to proper types
        df = df.astype({
            'year': 'int32',
            'month': 'int32',
            'ddd_demand': 'float32',
            'avg_temp_max': 'float32',
            'avg_temp_min': 'float32',
            'avg_humidity': 'float32',
            'total_precipitation': 'float32',
            'total_sunshine_hours': 'float32'
        })
        
        # Create date column and sort
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values('date')
        
        # Get last 12 months
        if len(df) < 12:
            logger.warning(f"Only {len(df)} months of data available")
            return df.to_csv(index=False)
        
        last_12 = df.iloc[-12:].copy()
        logger.info(f"Returning last 12 months of data (from {last_12['date'].min()} to {last_12['date'].max()})")
        
        return last_12[expected_cols].to_csv(index=False)
    except Exception as e:
        logger.error(f"Failed to load input data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load input data")

@app.get("/api/test-metrics")
async def get_test_metrics():
    """Get model test metrics"""
    try:
        filepath = os.path.join(DATA_DIR, 'test_metrics.csv')
        df = pd.read_csv(filepath)
        
        # Basic validation
        required_cols = ['model_name', 'mae']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Test metrics missing required columns")
        
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to load test metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load test metrics")

@app.get("/api/predictions")
async def get_predictions():
    """Get model predictions"""
    try:
        filepath = os.path.join(DATA_DIR, 'predictions.csv')
        df = pd.read_csv(filepath)
        
        # Basic validation
        required_cols = ['model_name', 'date', 'predicted_demand']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Predictions missing required columns")
        
        # Convert dates to consistent format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%B %Y')
        
        # Structure predictions by model
        predictions = {}
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            predictions[model] = dict(zip(model_data['date'], model_data['predicted_demand']))
        
        # Get best model from test metrics
        metrics_file = os.path.join(DATA_DIR, 'test_metrics.csv')
        metrics_df = pd.read_csv(metrics_file)
        best_model = metrics_df.loc[metrics_df['mae'].idxmin(), 'model_name']
        
        # Get last 6 months of actual demand
        input_file = os.path.join(DATA_DIR, 'raw_climate_ddd_merged_data.csv')
        input_df = pd.read_csv(input_file)
        input_df['date'] = pd.to_datetime(input_df[['year', 'month']].assign(day=1))
        last_6 = input_df.sort_values('date').tail(6)[['date', 'ddd_demand']]
        last_6['date'] = last_6['date'].dt.strftime('%Y-%m-%d')
        
        return {
            'predictions': predictions,
            'best_model': best_model,
            'test_maes': dict(zip(metrics_df['model_name'], metrics_df['mae'])),
            'last_6_months': last_6.to_dict('records'),
            'model_names': list(predictions.keys())
        }
    except Exception as e:
        logger.error(f"Failed to load predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load predictions")

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
