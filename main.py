import os
import logging
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from datetime import datetime
from typing import Dict, List, Any

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

# Expected CSV files
REQUIRED_FILES = [
    'raw_climate_ddd_merged_data.csv',
    'test_metrics.csv'
]

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv("POSTGRES_DB", "prod_db"),
    'user': os.getenv("POSTGRES_USER", "airflow"),
    'password': os.getenv("POSTGRES_PASSWORD", "airflow"),
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': os.getenv("POSTGRES_PORT", "5432")
}

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Successfully connected to PostgreSQL database")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@app.on_event("startup")
async def startup_event():
    """Validate required files and database table"""
    try:
        # Create directories if they don't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATIC_DIR, exist_ok=True)
        
        # Validate required CSV files
        for filename in REQUIRED_FILES:
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                logger.error(f"Required file not found: {filepath}")
                raise FileNotFoundError(f"Required file not found: {filepath}")
            
            # Test reading each file
            try:
                pd.read_csv(filepath)
                logger.info(f"Successfully validated file: {filename}")
            except Exception as e:
                logger.error(f"Failed to read {filename}: {str(e)}")
                raise ValueError(f"Failed to read {filename}: {str(e)}")
        
        # Validate database connection and ddd_predictions table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'ddd_predictions'
        """)
        if not cursor.fetchone():
            logger.error("Required table not found: ddd_predictions")
            raise ValueError("Required table not found: ddd_predictions")
        
        cursor.close()
        conn.close()
        logger.info("Startup validation completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to dashboard"""
    logger.info("Redirecting to dashboard")
    return RedirectResponse(url="/static/index.html")

@app.get("/api/input-data")
async def get_input_data():
    """Get the most recent 12 months of input data"""
    try:
        filepath = os.path.join(DATA_DIR, 'raw_climate_ddd_merged_data.csv')
        df = pd.read_csv(filepath)
        
        # Validate columns
        expected_cols = ['year', 'month', 'ddd_demand', 'avg_temp_max', 
                        'avg_temp_min', 'avg_humidity', 'total_precipitation', 
                        'total_sunshine_hours']
        
        if not all(col in df.columns for col in expected_cols):
            logger.error("Input data missing required columns")
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
            return JSONResponse(content=df.to_dict('records'))
        
        last_12 = df.iloc[-12:].copy()
        logger.info(f"Returning last 12 months of data (from {last_12['date'].min()} to {last_12['date'].max()})")
        
        return JSONResponse(content=last_12[expected_cols].to_dict('records'))
    except Exception as e:
        logger.error(f"Failed to load input data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load input data")

@app.get("/api/test-metrics")
async def get_test_metrics():
    """Get model test metrics"""
    try:
        filepath = os.path.join(DATA_DIR, 'test_metrics.csv')
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'model': 'model_name',
            'Model': 'model_name',
            'mean_absolute_error': 'mae',
            'MAE': 'mae'
        }
        df = df.rename(columns=column_mapping)
        
        # Validate columns
        required_cols = ['model_name', 'mae']
        if not all(col in df.columns for col in required_cols):
            logger.error("Test metrics missing required columns")
            raise ValueError("Test metrics missing required columns")
        
        # Convert to numeric
        df['mae'] = pd.to_numeric(df['mae'], errors='coerce')
        
        logger.info(f"Loaded test metrics for {len(df)} models")
        return JSONResponse(content=df.to_dict('records'))
    except Exception as e:
        logger.error(f"Failed to load test metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load test metrics")

@app.get("/api/predictions")
async def get_predictions():
    """Get model predictions from PostgreSQL and historical data from CSV"""
    try:
        # Load predictions from PostgreSQL
        conn = get_db_connection()
        query_predictions = """
            SELECT model, date, predicted_demand, generated_on
            FROM (
                SELECT model, date, predicted_demand, generated_on,
                       ROW_NUMBER() OVER (PARTITION BY model, date ORDER BY generated_on DESC) as rn
                FROM ddd_predictions
            ) t
            WHERE rn = 1
            ORDER BY model, date
        """
        df_predictions = pd.read_sql(query_predictions, conn)
        logger.info(f"Retrieved {len(df_predictions)} prediction rows from PostgreSQL")
        conn.close()
        
        # Load test metrics from CSV
        metrics_file = os.path.join(DATA_DIR, 'test_metrics.csv')
        metrics_df = pd.read_csv(metrics_file)
        metrics_df = metrics_df.rename(columns={
            'model': 'model_name',
            'Model': 'model_name',
            'mean_absolute_error': 'mae',
            'MAE': 'mae'
        })
        
        # Load last 6 months of historical data from CSV
        input_file = os.path.join(DATA_DIR, 'raw_climate_ddd_merged_data.csv')
        input_df = pd.read_csv(input_file)
        input_df['date'] = pd.to_datetime(input_df[['year', 'month']].assign(day=1))
        last_6 = input_df.sort_values('date').tail(6)[['date', 'ddd_demand']]
        last_6['date'] = last_6['date'].dt.strftime('%Y-%m-%d')
        
        # Structure predictions by model with properly formatted dates
        predictions = {}
        if not df_predictions.empty:
            df_predictions['date'] = pd.to_datetime(df_predictions['date']).dt.strftime('%B %Y')
            for model in df_predictions['model'].unique():
                model_data = df_predictions[df_predictions['model'] == model]
                predictions[model] = dict(zip(model_data['date'], model_data['predicted_demand']))
        
        # Get best model from test metrics
        best_model = None
        if not metrics_df.empty and 'mae' in metrics_df.columns:
            best_model = metrics_df.loc[metrics_df['mae'].idxmin(), 'model_name'] if not metrics_df['mae'].isnull().all() else None
        
        return {
            "predictions": predictions,
            "best_model": best_model,
            "test_maes": dict(zip(metrics_df['model_name'], metrics_df['mae'])),
            "last_6_months": last_6.to_dict('records'),
            "model_names": list(predictions.keys())
        }
    except Exception as e:
        logger.error(f"Failed to load predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
