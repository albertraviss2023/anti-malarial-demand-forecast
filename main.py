import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from datetime import datetime
import traceback
from starlette.responses import JSONResponse
import time

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Anti-Malarial Demand Forecasting Dashboard",
    description="Dashboard for anti-malarial demand forecasting"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host
    logger.debug(f"Request: {request.method} {request.url.path} - Client IP: {client_ip} - Headers: {dict(request.headers)}")
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000  # ms
        logger.debug(f"Response: {request.url.path} - Status: {response.status_code} - Client IP: {client_ip} - Duration: {duration:.2f}ms")
        return response
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Error processing {request.method} {request.url.path} - Client IP: {client_ip} - Duration: {duration:.2f}ms: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Expected columns for selected_data.csv
EXPECTED_COLS = [
    'avg_temp_max', 'avg_temp_min', 'avg_humidity',
    'total_precipitation', 'total_sunshine_hours', 'ddd_demand'
]

@app.on_event("startup")
async def startup_event():
    """Validate required files and directories"""
    logger.debug("Starting application")
    try:
        if not os.path.exists(DATA_DIR):
            logger.error(f"Data directory not found: {DATA_DIR}")
            raise FileNotFoundError(f"Data directory {DATA_DIR} not found")
        logger.debug(f"Data directory exists: {DATA_DIR}")
        
        if not os.path.exists(STATIC_DIR):
            logger.warning(f"Static directory not found, creating: {STATIC_DIR}")
            os.makedirs(STATIC_DIR)
        logger.debug(f"Static directory exists: {STATIC_DIR}")
        
        required_files = ['selected_data.csv', 'test_metrics.csv', 'predictions.csv']
        for file in required_files:
            file_path = os.path.join(DATA_DIR, file)
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"File {file_path} not found")
            logger.debug(f"Found required file: {file_path}")
        
        # Validate selected_data.csv row count
        data_path = os.path.join(DATA_DIR, 'selected_data.csv')
        try:
            df = pd.read_csv(data_path)
            row_count = len(df)
            logger.debug(f"selected_data.csv: {row_count} data rows, columns: {df.columns.tolist()}")
            if row_count != 99:
                logger.error(f"Expected 99 data rows in selected_data.csv, got {row_count}")
                raise ValueError(f"Expected 99 data rows, got {row_count}")
            if not all(col in df.columns for col in EXPECTED_COLS):
                logger.error(f"Expected columns {EXPECTED_COLS}, got {df.columns.tolist()}")
                raise ValueError(f"Expected columns {EXPECTED_COLS}")
            logger.debug(f"First row: {df.iloc[0].to_dict()}")
        except Exception as e:
            logger.error(f"Failed to validate selected_data.csv: {str(e)}")
            raise
        
        # Validate other CSVs
        for file in ['test_metrics.csv', 'predictions.csv']:
            file_path = os.path.join(DATA_DIR, file)
            try:
                df = pd.read_csv(file_path)
                logger.debug(f"Validated {file}: {len(df)} rows, columns: {df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Failed to read {file}: {str(e)}")
                raise
        
        logger.info("Startup validation completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}\n{traceback.format_exc()}")
        raise

# Root redirect to dashboard
@app.get("/", summary="Redirect to dashboard")
async def root():
    logger.debug("Redirecting root to /static/index.html")
    return RedirectResponse(url="/static/index.html")

# Mount static files at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
logger.debug(f"Mounted static directory at /static: {STATIC_DIR}")

@app.get("/api/input-data", summary="Get last 12 months of input data")
async def get_input_data():
    """Serve last 12 months of selected_data.csv with assigned dates"""
    logger.debug("Received request for /api/input-data")
    try:
        data_path = os.path.join(DATA_DIR, 'selected_data.csv')
        logger.debug(f"Loading CSV: {data_path}")
        df = pd.read_csv(data_path)
        row_count = len(df)
        logger.debug(f"Loaded selected_data.csv: {row_count} data rows, columns: {df.columns.tolist()}")
        if row_count != 99:
            logger.error(f"Expected 99 data rows in selected_data.csv, got {row_count}")
            raise ValueError(f"Expected 99 data rows, got {row_count}")
        
        # Log sample data
        logger.debug(f"First row: {df.iloc[0].to_dict()}")
        logger.debug(f"Data types: {df.dtypes.to_dict()}")
        
        # Validate columns
        if not all(col in df.columns for col in EXPECTED_COLS):
            logger.error(f"Expected columns {EXPECTED_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Expected columns {EXPECTED_COLS}")
        
        # Validate data
        df = df.astype({col: 'float32' for col in EXPECTED_COLS})
        if df.isnull().any().any():
            logger.warning("Missing values in input data, filling with column means")
            df.fillna(df.mean(), inplace=True)
        if (df['avg_humidity'] < 0).any() or (df['avg_humidity'] > 100).any():
            logger.error("Invalid humidity values: must be between 0 and 100")
            raise ValueError("Invalid humidity values")
        if (df['total_precipitation'] < 0).any() or (df['total_sunshine_hours'] < 0).any():
            logger.error("Negative precipitation or sunshine hours detected")
            raise ValueError("Negative precipitation or sunshine hours")
        if (df['ddd_demand'] < 0).any():
            logger.error("Negative demand values detected")
            raise ValueError("Negative demand values")
        logger.debug("Data validation passed")
        
        # Assign dates (Jan 2017 to Mar 2025)
        start_date = pd.to_datetime('2017-01-01')
        dates = pd.date_range(start=start_date, periods=99, freq='MS')
        df['date'] = dates
        last_12 = df.iloc[-12:].copy()
        logger.debug(f"Selected last 12 rows: {last_12['date'].dt.strftime('%Y-%m-%d').tolist()}")
        
        # Format date as YYYY-MM-DD
        last_12['date'] = last_12['date'].dt.strftime('%Y-%m-%d')
        
        # Calculate stats
        stats = {
            'max_temp_avg': float(last_12['avg_temp_max'].mean()),
            'min_temp_avg': float(last_12['avg_temp_min'].mean()),
            'humidity_avg': float(last_12['avg_humidity'].mean()),
            'precip_total': float(last_12['total_precipitation'].sum()),
            'sunshine_total': float(last_12['total_sunshine_hours'].sum()),
            'current_demand': float(last_12['ddd_demand'].iloc[-1])
        }
        logger.debug(f"Computed stats: {stats}")
        
        response = {
            'data': last_12.to_dict(orient='records'),
            'stats': stats
        }
        logger.info(f"Returning input data: {len(response['data'])} rows, stats keys: {list(stats.keys())}")
        logger.debug(f"Response sample: {response['data'][:1]}")
        return response
    except Exception as e:
        logger.error(f"Failed to load input data: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load input data: {str(e)}")

@app.get("/api/test-metrics", summary="Get model test metrics")
async def get_test_metrics():
    """Serve test_metrics.csv"""
    logger.debug("Received request for /api/test-metrics")
    try:
        metrics_path = os.path.join(DATA_DIR, 'test_metrics.csv')
        logger.debug(f"Loading CSV: {metrics_path}")
        df = pd.read_csv(metrics_path)
        logger.debug(f"Loaded test_metrics.csv: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Validate columns
        required_cols = ['model_name', 'mae']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Expected columns {required_cols}, got {df.columns.tolist()}")
            raise ValueError(f"Expected columns {required_cols}")
        if (df['mae'] < 0).any():
            logger.error("Negative MAE values detected")
            raise ValueError("Negative MAE values")
        
        # Log sample data
        logger.debug(f"First row: {df.iloc[0].to_dict()}")
        
        response = df.to_dict(orient='records')
        logger.info(f"Returning test metrics: {len(response)} rows")
        logger.debug(f"Response sample: {response[:1]}")
        return response
    except Exception as e:
        logger.error(f"Failed to load test metrics: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load test metrics: {str(e)}")

@app.get("/api/predictions", summary="Get model predictions")
async def get_predictions():
    """Serve predictions.csv formatted for dashboard"""
    logger.debug("Received request for /api/predictions")
    try:
        predictions_path = os.path.join(DATA_DIR, 'predictions.csv')
        logger.debug(f"Loading CSV: {predictions_path}")
        df = pd.read_csv(predictions_path)
        logger.debug(f"Loaded predictions.csv: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Validate columns
        required_cols = ['model_name', 'date', 'predicted_demand']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Expected columns {required_cols}, got {df.columns.tolist()}")
            raise ValueError(f"Expected columns {required_cols}")
        
        # Log sample data
        logger.debug(f"First row: {df.iloc[0].to_dict()}")
        logger.debug(f"Raw dates: {df['date'].tolist()}")
        
        # Convert and validate dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isnull().any():
            logger.error("Invalid date formats in predictions.csv")
            raise ValueError("Invalid date formats in predictions.csv")
        
        # Format dates as 'MMMM yyyy'
        df['date'] = df['date'].dt.strftime('%B %Y')
        logger.debug(f"Formatted dates: {df['date'].tolist()}")
        
        # Validate predicted_demand
        df['predicted_demand'] = pd.to_numeric(df['predicted_demand'], errors='coerce')
        if df['predicted_demand'].isnull().any():
            logger.error("Invalid predicted_demand values")
            raise ValueError("Invalid predicted_demand values")
        
        # Structure predictions
        predictions = {}
        model_names = df['model_name'].unique()
        expected_dates = ['April 2025', 'May 2025', 'June 2025']
        logger.debug(f"Model names: {model_names.tolist()}")
        for model in model_names:
            model_preds = df[df['model_name'] == model][['date', 'predicted_demand']]
            model_dict = {row['date']: float(row['predicted_demand']) for _, row in model_preds.iterrows()}
            if not all(date in model_dict for date in expected_dates):
                logger.warning(f"Missing expected dates for model {model}: {expected_dates}")
            predictions[model] = model_dict
        logger.debug(f"Predictions structure: {predictions}")
        
        # Load test metrics for best_model
        metrics_path = os.path.join(DATA_DIR, 'test_metrics.csv')
        logger.debug(f"Loading test metrics for best model: {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)
        test_maes = dict(zip(metrics_df['model_name'], metrics_df['mae']))
        best_model = min(test_maes, key=test_maes.get) if test_maes else None
        logger.debug(f"Test MAEs: {test_maes}, Best model: {best_model}")
        
        # Load last 6 months of demand
        data_path = os.path.join(DATA_DIR, 'selected_data.csv')
        logger.debug(f"Loading input data for last 6 months: {data_path}")
        input_df = pd.read_csv(data_path)
        if len(input_df) != 99:
            logger.error(f"Expected 99 data rows in selected_data.csv, got {len(input_df)}")
            raise ValueError(f"Expected 99 data rows, got {len(input_df)}")
        start_date = pd.to_datetime('2017-01-01')
        dates = pd.date_range(start=start_date, periods=99, freq='MS')
        input_df['date'] = dates
        last_6 = input_df.iloc[-6:][['date', 'ddd_demand']].copy()
        last_6['date'] = last_6['date'].dt.strftime('%Y-%m-%d')
        logger.debug(f"Last 6 months dates: {last_6['date'].tolist()}")
        
        response = {
            'predictions': predictions,
            'best_model': best_model,
            'test_maes': test_maes,
            'last_6_months': last_6.to_dict(orient='records'),
            'model_names': list(model_names)
        }
        logger.info(f"Returning predictions: {len(response['predictions'])} models, best_model: {response['best_model']}")
        logger.debug(f"Response sample: {response['predictions']}")
        return response
    except Exception as e:
        logger.error(f"Failed to load predictions: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

# Handle 404 errors
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    client_ip = request.client.host
    logger.error(f"404 Not Found: {request.method} {request.url.path} - Client IP: {client_ip} - Headers: {dict(request.headers)}")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})
