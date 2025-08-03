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
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATIC_DIR, exist_ok=True)
        
        for filename in REQUIRED_FILES:
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                logger.error(f"Required file not found: {filepath}")
                raise FileNotFoundError(f"Required file not found: {filepath}")
            
            try:
                pd.read_csv(filepath)
            except Exception as e:
                logger.error(f"Failed to read {filename}: {str(e)}")
                raise ValueError(f"Failed to read {filename}: {str(e)}")
        
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
    return RedirectResponse(url="/static/index.html")

@app.get("/api/input-data")
async def get_input_data():
    try:
        filepath = os.path.join(DATA_DIR, 'raw_climate_ddd_merged_data.csv')
        df = pd.read_csv(filepath)
        
        expected_cols = ['year', 'month', 'ddd_demand', 'avg_temp_max', 
                        'avg_temp_min', 'avg_humidity', 'total_precipitation', 
                        'total_sunshine_hours']
        
        if not all(col in df.columns for col in expected_cols):
            raise ValueError("Input data missing required columns")
        
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
        
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values('date')
        
        if len(df) < 12:
            logger.warning(f"Only {len(df)} months of data available")
            return JSONResponse(content=df.to_dict('records'))
        
        last_12 = df.iloc[-12:].copy()
        last_12['date'] = last_12['date'].dt.strftime('%Y-%m-%d')
        return JSONResponse(content=last_12[expected_cols + ['date']].to_dict('records'))
    except Exception as e:
        logger.error(f"Failed to load input data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load input data")

@app.get("/api/test-metrics")
async def get_test_metrics():
    try:
        filepath = os.path.join(DATA_DIR, 'test_metrics.csv')
        df = pd.read_csv(filepath)
        
        # Normalize model names to lowercase
        df['model_name'] = df.iloc[:, 0].str.strip().str.lower()
        df['mae'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        
        if df.empty or df['mae'].isnull().all():
            raise ValueError("No valid test metrics found")
        
        return JSONResponse(content=df[['model_name', 'mae']].to_dict('records'))
    except Exception as e:
        logger.error(f"Failed to load test metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load test metrics")

@app.get("/api/predictions")
async def get_predictions():
    try:
        conn = get_db_connection()
        query = """
            SELECT model, date, predicted_demand, generated_on
            FROM (
                SELECT model, date, predicted_demand, generated_on,
                       ROW_NUMBER() OVER (PARTITION BY LOWER(model), date ORDER BY generated_on DESC) as rn
                FROM ddd_predictions
            ) t
            WHERE rn = 1
            ORDER BY model, date
        """
        df_predictions = pd.read_sql(query, conn)
        conn.close()
        
        # Normalize model names to lowercase
        df_predictions['model'] = df_predictions['model'].str.lower()
        
        # Load test metrics
        metrics_file = os.path.join(DATA_DIR, 'test_metrics.csv')
        metrics_df = pd.read_csv(metrics_file)
        metrics_df['model_name'] = metrics_df.iloc[:, 0].str.strip().str.lower()
        metrics_df['mae'] = pd.to_numeric(metrics_df.iloc[:, 2], errors='coerce')
        
        # Load historical data
        input_file = os.path.join(DATA_DIR, 'raw_climate_ddd_merged_data.csv')
        input_df = pd.read_csv(input_file)
        input_df['date'] = pd.to_datetime(input_df[['year', 'month']].assign(day=1))
        last_6 = input_df.sort_values('date').tail(6)[['date', 'ddd_demand']]
        last_6['date'] = last_6['date'].dt.strftime('%Y-%m-%d')
        
        # Structure predictions
        predictions = {}
        if not df_predictions.empty:
            df_predictions['date'] = pd.to_datetime(df_predictions['date']).dt.strftime('%B %Y')
            for model in df_predictions['model'].unique():
                model_data = df_predictions[df_predictions['model'] == model]
                predictions[model] = dict(zip(
                    model_data['date'],
                    model_data['predicted_demand'].round(2)
                ))
        
        # Get best model
        best_model = None
        if not metrics_df.empty:
            valid_metrics = metrics_df.dropna(subset=['mae'])
            if not valid_metrics.empty:
                best_model = valid_metrics.loc[valid_metrics['mae'].idxmin(), 'model_name']
        
        return {
            "predictions": predictions,
            "best_model": best_model,
            "test_maes": dict(zip(metrics_df['model_name'], metrics_df['mae'].round(4))),
            "last_6_months": last_6.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Failed to load predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")