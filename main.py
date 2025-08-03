import os
import logging
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    title="Uganda Malaria Dashboard API",
    description="API for the Uganda Malaria Surveillance Dashboard",
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
    'districts.csv',
    'test_maes.csv',
    'malaria_historical.csv'
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
        
        # Validate database connection and malaria_predictions table
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'malaria_predictions'
            """)
            if not cursor.fetchone():
                logger.error("Required table not found: malaria_predictions")
                raise ValueError("Required table not found: malaria_predictions")
            
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Database validation failed: {str(e)}")
            raise
        
        logger.info("Startup validation completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/")
async def serve_dashboard():
    """Serve the dashboard"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(index_path)

@app.get("/api/districts")
async def get_districts():
    """Get district data from CSV"""
    try:
        filepath = os.path.join(DATA_DIR, "districts.csv")
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist
        required_cols = ['district', 'latitude', 'longitude']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Districts CSV missing required columns: {required_cols}")
            
        logger.info(f"Loaded {len(df)} districts from districts.csv")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to load districts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load districts: {str(e)}")

@app.get("/api/malaria_predictions")
async def get_malaria_predictions():
    """Get model predictions from PostgreSQL for the best model"""
    try:
        # Load test metrics from CSV to determine best model
        maes_file = os.path.join(DATA_DIR, "test_maes.csv")
        maes_df = pd.read_csv(maes_file)
        
        # Validate test metrics columns
        required_cols = ['model_name', 'mae_sum']
        if not all(col in maes_df.columns for col in required_cols):
            raise ValueError("Test metrics missing required columns")
        
        # Get best model (lowest MAE)
        if maes_df.empty:
            raise ValueError("No model metrics available")
            
        best_model = maes_df.loc[maes_df['mae_sum'].idxmin()]['model_name']
        logger.info(f"Best model identified: {best_model}")
        
        # Load predictions from PostgreSQL
        conn = None
        try:
            conn = get_db_connection()
            query = """
                SELECT district, model, month, predicted_mal_cases
                FROM (
                    SELECT district, model, month, predicted_mal_cases,
                           ROW_NUMBER() OVER (PARTITION BY district, model, month ORDER BY generated_on DESC) as rn
                    FROM malaria_predictions
                    WHERE model = %s
                ) t
                WHERE rn = 1
                ORDER BY district, month
            """
            df = pd.read_sql(query, conn, params=(best_model,))
            
            if df.empty:
                logger.warning(f"No predictions found for model {best_model}")
                return []
                
            # Pivot the data to have months as columns
            pivoted = df.pivot(index="district", columns="month", values="predicted_mal_cases").reset_index()
            
            # Ensure we have all expected months
            expected_months = ["May", "June", "August"]
            for month in expected_months:
                if month not in pivoted.columns:
                    pivoted[month] = 0.0
            
            # Convert to the required format
            threshold = 10000  # Adjust this threshold as needed
            results = []
            for _, row in pivoted.iterrows():
                district_data = {
                    "district": row["district"],
                    "cases": {
                        "May": float(row.get("May", 0)),
                        "June": float(row.get("June", 0)),
                        "August": float(row.get("August", 0))
                    },
                    "status": {
                        "May": "high" if row.get("May", 0) > threshold else "medium" if row.get("May", 0) > threshold * 0.7 else "low",
                        "June": "high" if row.get("June", 0) > threshold else "medium" if row.get("June", 0) > threshold * 0.7 else "low",
                        "August": "high" if row.get("August", 0) > threshold else "medium" if row.get("August", 0) > threshold * 0.7 else "low"
                    }
                }
                results.append(district_data)
            
            logger.info(f"Returning predictions for {len(results)} districts")
            return results
            
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Failed to load malaria predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

@app.get("/api/model_metrics")
async def get_model_metrics():
    """Get model test metrics from CSV"""
    try:
        filepath = os.path.join(DATA_DIR, "test_maes.csv")
        df = pd.read_csv(filepath)
        
        required_cols = ['model_name', 'mae_sum', 'mae', 't+1_mae', 't+2_mae', 't+3_mae']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Test metrics missing required columns")
            
        logger.info(f"Loaded model metrics for {len(df)} models")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to load model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model metrics: {str(e)}")

@app.get("/api/malaria_historical")
async def get_malaria_historical():
    """Get historical malaria and climate data from CSV"""
    try:
        filepath = os.path.join(DATA_DIR, "malaria_historical.csv")
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['year', 'month', 'mal_cases', 'avg_temp_max', 
                        'avg_temp_min', 'avg_humidity', 'sum_precipitation']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Historical data missing required columns")
        
        # Create date string
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m')
        
        # Aggregate malaria cases by month
        malaria_agg = df.groupby(['year', 'month', 'date'])['mal_cases'].sum().reset_index()
        malaria_data = malaria_agg[['date', 'mal_cases']].rename(columns={'mal_cases': 'cases'}).to_dict(orient="records")
        
        # Aggregate climate data by month
        climate_agg = df.groupby(['year', 'month', 'date']).agg({
            'avg_temp_max': 'mean',
            'avg_temp_min': 'mean',
            'avg_humidity': 'mean',
            'sum_precipitation': 'sum'
        }).reset_index()
        climate_data = climate_agg.to_dict(orient="records")
        
        # Calculate correlations between climate factors and malaria cases
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
        
        correlation_data = corr_data.to_dict(orient="records")
        
        logger.info("Successfully loaded historical data")
        return {
            "malaria_cases": malaria_data,
            "climate": climate_data,
            "correlations": correlation_data,
            "correlation_coeffs": correlation_coeffs
        }
    except Exception as e:
        logger.error(f"Failed to load historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load historical data: {str(e)}")

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
