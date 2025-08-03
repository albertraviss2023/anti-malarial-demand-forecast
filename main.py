import os
import glob
import re
import logging
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --------------------------
# Configuration and Setup
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = "/opt/airflow/results"
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv("POSTGRES_DB", "prod_db"),
    'user': os.getenv("POSTGRES_USER", "airflow"),
    'password': os.getenv("POSTGRES_PASSWORD", "airflow"),
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': os.getenv("POSTGRES_PORT", "5432")
}

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malaria_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Uganda Malaria Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --------------------------
# Data Loading Functions
# --------------------------
def standardize_model_name(model_name):
    """Standardize model names to lowercase and handle known variations"""
    model_name = str(model_name).strip().lower()
    if model_name in ['transformer', 'transformers']:
        return 'transformer'
    return model_name

from datetime import datetime

def get_recent_prediction_months():
    """Get the 3 most recent prediction months from the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT month FROM malaria_predictions
        """)
        months = [row[0] for row in cur.fetchall()]
        
        if not months:
            logger.warning("No months found in the database.")
            return []

        # Convert month strings (e.g., 'September_2025') to datetime for sorting
        def parse_month(month_str):
            try:
                return datetime.strptime(month_str, '%B_%Y')
            except ValueError as e:
                logger.error(f"Invalid month format: {month_str}, error: {str(e)}")
                return None

        # Filter out invalid months and sort by date
        parsed_months = [(month, parse_month(month)) for month in months]
        valid_months = [month for month, parsed in parsed_months if parsed is not None]
        sorted_months = sorted(valid_months, key=lambda x: parse_month(x), reverse=True)
        
        # Return the 3 most recent months
        recent_months = sorted_months[:3]
        logger.info(f"Retrieved recent months: {recent_months}")
        return recent_months if recent_months else []

    except Exception as e:
        logger.error(f"Error getting recent months: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()
            
# --------------------------
# API Endpoints
# --------------------------
@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/districts")
async def get_districts():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "districts.csv"))
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to load districts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model_metrics")
async def get_model_metrics():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "test_maes.csv"))
        df['model_name'] = df['model_name'].str.lower().str.strip()
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to load model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/malaria_predictions")
async def get_malaria_predictions():
    try:
        # Get the 3 most recent prediction months
        months = get_recent_prediction_months()
        if not months:
            logger.warning("No prediction months available in the database.")
            raise HTTPException(status_code=404, detail="No prediction months available in the database.")

        # Get best model from metrics
        maes_df = pd.read_csv(os.path.join(DATA_DIR, "test_maes.csv"))
        maes_df['model_name'] = maes_df['model_name'].str.lower().str.strip()
        best_model = maes_df.loc[maes_df['mae_sum'].idxmin()]['model_name']
        logger.info(f"Using best model: {best_model} for months: {months}")

        # Get predictions from database
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
            SELECT district, month, predicted_mal_cases
            FROM (
                SELECT district, month, predicted_mal_cases,
                       ROW_NUMBER() OVER (
                           PARTITION BY district, month 
                           ORDER BY generated_on DESC
                       ) as rn
                FROM malaria_predictions
                WHERE LOWER(TRIM(model)) = %s
                AND month IN %s
            ) t
            WHERE rn = 1
        """
        df = pd.read_sql(query, conn, params=(best_model, tuple(months)))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No predictions found for model {best_model} in months {months}")

        # Format response
        pivoted = df.pivot(index="district", columns="month", values="predicted_mal_cases").fillna(0)
        threshold = 10000
        results = []
        
        for district, row in pivoted.iterrows():
            cases = {month: float(row.get(month, 0)) for month in months}
            status = {
                month: "high" if row.get(month, 0) > threshold 
                      else "medium" if row.get(month, 0) > threshold * 0.7 
                      else "low"
                for month in months
            }
            results.append({
                "district": district,
                "cases": cases,
                "status": status
            })

        logger.info(f"Returned predictions for {len(results)} districts")
        return {
            "predictions": results,
            "months": months  # Include months in response for frontend
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/api/malaria_historical")
async def get_malaria_historical():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "malaria_historical.csv"))
        
        # Create date column
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m')
        
        # Aggregate malaria cases
        malaria_agg = df.groupby(['year', 'month', 'date'])['mal_cases'].sum().reset_index()
        malaria_data = malaria_agg[['date', 'mal_cases']].rename(columns={'mal_cases': 'cases'}).to_dict(orient="records")
        
        # Aggregate climate data
        climate_agg = df.groupby(['year', 'month', 'date']).agg({
            'avg_temp_max': 'mean',
            'avg_temp_min': 'mean',
            'avg_humidity': 'mean',
            'sum_precipitation': 'sum'
        }).reset_index()
        climate_data = climate_agg.to_dict(orient="records")
        
        # Calculate correlations
        corr_data = df.groupby(['year', 'month']).agg({
            'mal_cases': 'sum',
            'avg_temp_max': 'mean',
            'avg_temp_min': 'mean',
            'avg_humidity': 'mean',
            'sum_precipitation': 'sum'
        }).reset_index()
        
        correlation_coeffs = {
            'avg_temp_max': corr_data['mal_cases'].corr(corr_data['avg_temp_max']),
            'avg_temp_min': corr_data['mal_cases'].corr(corr_data['avg_temp_min']),
            'avg_humidity': corr_data['mal_cases'].corr(corr_data['avg_humidity']),
            'sum_precipitation': corr_data['mal_cases'].corr(corr_data['sum_precipitation'])
        }
        
        return {
            "malaria_cases": malaria_data,
            "climate": climate_data,
            "correlations": corr_data.to_dict(orient="records"),
            "correlation_coeffs": correlation_coeffs
        }
    except Exception as e:
        logger.error(f"Failed to load historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load data when script runs directly
    load_predictions_to_db()
else:
    # When running as FastAPI app, ensure tables exist
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS malaria_predictions (
                district TEXT NOT NULL,
                model TEXT NOT NULL,
                month TEXT NOT NULL,
                predicted_mal_cases FLOAT NOT NULL,
                generated_on DATE NOT NULL,
                PRIMARY KEY (district, model, month, generated_on)
            );
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
