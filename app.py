import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from datetime import datetime
import traceback
import time

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Malaria Dashboard Suite",
    description="Unified dashboard for anti-malarial demand forecasting and malaria case mapping"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host
    logger.debug(f"Request: {request.method} {request.url.path} - Client IP: {client_ip} - Headers: {dict(request.headers)}")
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        logger.debug(f"Response: {request.url.path} - Status: {response.status_code} - Client IP: {client_ip} - Duration: {duration:.2f}ms")
        return response
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Error processing {request.method} {request.url.path} - Client IP: {client_ip} - Duration: {duration:.2f}ms: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

DEMAND_COLS = ['avg_temp_max', 'avg_temp_min', 'avg_humidity', 'total_precipitation', 'total_sunshine_hours', 'ddd_demand']
TEST_METRICS_COLS = ['model_name', 'mae']
PREDICTIONS_COLS = ['model_name', 'date', 'predicted_demand']
MAE_COLS = ['model_name', 'mae_sum', 't+1_mae', 't+2_mae', 't+3_mae']
MALARIA_PRED_COLS = ['district', 'model', 'month', 'predicted_mal_cases']
HISTORICAL_COLS = ['year', 'month', 'district', 'mal_cases']

@app.on_event("startup")
async def startup_event():
    logger.debug("Starting application")
    try:
        if not os.path.exists(DATA_DIR):
            logger.error(f"Data directory not found: {DATA_DIR}")
            raise FileNotFoundError(f"Data directory {DATA_DIR} not found")
        if not os.path.exists(STATIC_DIR):
            logger.warning(f"Static directory not found, creating: {STATIC_DIR}")
            os.makedirs(STATIC_DIR)

        required_files = [
            'selected_data.csv', 'test_metrics.csv', 'predictions.csv',
            'test_maes.csv', 'predictions_may_june_july_2025.csv', 'malaria_historical.csv'
        ]
        for file in required_files:
            file_path = os.path.join(DATA_DIR, file)
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"File {file} not found")
            logger.debug(f"Found file: {file_path}")

        data_path = os.path.join(DATA_DIR, 'selected_data.csv')
        df = pd.read_csv(data_path)
        logger.debug(f"Loaded selected_data.csv with {len(df)} rows")
        if len(df) != 99:
            logger.error(f"Expected 99 rows in selected_data.csv, got {len(df)}")
            raise ValueError(f"Expected 99 rows, got {len(df)}")
        if not all(col in df.columns for col in DEMAND_COLS):
            logger.error(f"Missing columns {DEMAND_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {DEMAND_COLS}")

        metrics_path = os.path.join(DATA_DIR, 'test_metrics.csv')
        df = pd.read_csv(metrics_path)
        logger.debug(f"Loaded test_metrics.csv with {len(df)} rows")
        if not all(col in df.columns for col in TEST_METRICS_COLS):
            logger.error(f"Missing columns {TEST_METRICS_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {TEST_METRICS_COLS}")

        pred_path = os.path.join(DATA_DIR, 'predictions.csv')
        df = pd.read_csv(pred_path)
        logger.debug(f"Loaded predictions.csv with {len(df)} rows")
        if not all(col in df.columns for col in PREDICTIONS_COLS):
            logger.error(f"Missing columns {PREDICTIONS_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {PREDICTIONS_COLS}")

        mae_path = os.path.join(DATA_DIR, 'test_maes.csv')
        df = pd.read_csv(mae_path)
        logger.debug(f"Loaded test_maes.csv with {len(df)} rows")
        if not all(col in df.columns for col in MAE_COLS):
            logger.error(f"Missing columns {MAE_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {MAE_COLS}")

        mal_pred_path = os.path.join(DATA_DIR, 'predictions_may_june_july_2025.csv')
        df = pd.read_csv(mal_pred_path)
        logger.debug(f"Loaded predictions_may_june_july_2025.csv with {len(df)} rows")
        if not all(col in df.columns for col in MALARIA_PRED_COLS):
            logger.error(f"Missing columns {MALARIA_PRED_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {MALARIA_PRED_COLS}")

        hist_path = os.path.join(DATA_DIR, 'malaria_historical.csv')
        df = pd.read_csv(hist_path)
        logger.debug(f"Loaded malaria_historical.csv with {len(df)} rows")
        if not all(col in df.columns for col in HISTORICAL_COLS):
            logger.error(f"Missing columns {HISTORICAL_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {HISTORICAL_COLS}")

        logger.info("Startup validation completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}\n{traceback.format_exc()}")
        raise

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
logger.debug(f"Mounted static directory at /static: {STATIC_DIR}")

@app.get("/", summary="Redirect to main dashboard")
async def root():
    logger.debug("Redirecting root to /static/index.html")
    return RedirectResponse(url="/static/index.html")

@app.get("/api/input-data", summary="Get last 12 months of input data")
async def get_input_data():
    logger.debug("Received request for /api/input-data")
    try:
        data_path = os.path.join(DATA_DIR, 'selected_data.csv')
        df = pd.read_csv(data_path)
        if len(df) != 99:
            logger.error(f"Expected 99 rows in selected_data.csv, got {len(df)}")
            raise ValueError(f"Expected 99 rows, got {len(df)}")
        if not all(col in df.columns for col in DEMAND_COLS):
            logger.error(f"Missing columns {DEMAND_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {DEMAND_COLS}")

        df = df.astype({col: 'float32' for col in DEMAND_COLS})
        if df.isnull().any().any():
            logger.warning("Missing values in input data, filling with column means")
            df.fillna(df.mean(), inplace=True)
        if (df['avg_humidity'] < 0).any() or (df['avg_humidity'] > 100).any():
            logger.error("Invalid humidity values")
            raise ValueError("Invalid humidity values")
        if (df['total_precipitation'] < 0).any() or (df['total_sunshine_hours'] < 0).any():
            logger.error("Negative precipitation or sunshine hours")
            raise ValueError("Negative precipitation or sunshine hours")
        if (df['ddd_demand'] < 0).any():
            logger.error("Negative demand values")
            raise ValueError("Negative demand values")

        start_date = pd.to_datetime('2017-01-01')
        dates = pd.date_range(start=start_date, periods=99, freq='MS')
        df['date'] = dates
        last_12 = df.iloc[-12:].copy()
        last_12['date'] = last_12['date'].dt.strftime('%Y-%m-%d')

        stats = {
            'max_temp_avg': float(last_12['avg_temp_max'].mean()),
            'min_temp_avg': float(last_12['avg_temp_min'].mean()),
            'humidity_avg': float(last_12['avg_humidity'].mean()),
            'precip_total': float(last_12['total_precipitation'].sum()),
            'sunshine_total': float(last_12['total_sunshine_hours'].sum()),
            'current_demand': float(last_12['ddd_demand'].iloc[-1])
        }

        response = {
            'data': last_12.to_dict(orient='records'),
            'stats': stats
        }
        logger.info(f"Returning input data: {len(response['data'])} rows")
        return response
    except Exception as e:
        logger.error(f"Failed to load input data: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load input data: {str(e)}")

@app.get("/api/test-metrics", summary="Get demand model test metrics")
async def get_test_metrics():
    logger.debug("Received request for /api/test-metrics")
    try:
        metrics_path = os.path.join(DATA_DIR, 'test_metrics.csv')
        df = pd.read_csv(metrics_path)
        if not all(col in df.columns for col in TEST_METRICS_COLS):
            logger.error(f"Missing columns {TEST_METRICS_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {TEST_METRICS_COLS}")
        if (df['mae'] < 0).any():
            logger.error("Negative MAE values detected")
            raise ValueError("Negative MAE values")

        response = df.to_dict(orient='records')
        logger.info(f"Returning test metrics: {len(response)} rows")
        return response
    except Exception as e:
        logger.error(f"Failed to load test metrics: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load test metrics: {str(e)}")

@app.get("/api/predictions", summary="Get demand model predictions")
async def get_predictions():
    logger.debug("Received request for /api/predictions")
    try:
        predictions_path = os.path.join(DATA_DIR, 'predictions.csv')
        df = pd.read_csv(predictions_path)
        if not all(col in df.columns for col in PREDICTIONS_COLS):
            logger.error(f"Missing columns {PREDICTIONS_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {PREDICTIONS_COLS}")

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isnull().any():
            logger.error("Invalid date formats in predictions.csv")
            raise ValueError("Invalid date formats")
        df['date'] = df['date'].dt.strftime('%B %Y')

        df['predicted_demand'] = pd.to_numeric(df['predicted_demand'], errors='coerce')
        if df['predicted_demand'].isnull().any():
            logger.error("Invalid predicted_demand values")
            raise ValueError("Invalid predicted_demand values")

        predictions = {}
        model_names = df['model_name'].unique()
        expected_dates = ['April 2025', 'May 2025', 'June 2025']
        for model in model_names:
            model_preds = df[df['model_name'] == model][['date', 'predicted_demand']]
            model_dict = {row['date']: float(row['predicted_demand']) for _, row in model_preds.iterrows()}
            predictions[model] = model_dict

        metrics_path = os.path.join(DATA_DIR, 'test_metrics.csv')
        metrics_df = pd.read_csv(metrics_path)
        test_maes = dict(zip(metrics_df['model_name'], metrics_df['mae']))
        best_model = min(test_maes, key=test_maes.get) if test_maes else None

        data_path = os.path.join(DATA_DIR, 'selected_data.csv')
        input_df = pd.read_csv(data_path)
        if len(input_df) != 99:
            logger.error(f"Expected 99 rows in selected_data.csv, got {len(input_df)}")
            raise ValueError(f"Expected 99 rows, got {len(input_df)}")
        start_date = pd.to_datetime('2017-01-01')
        dates = pd.date_range(start=start_date, periods=99, freq='MS')
        input_df['date'] = dates
        last_6 = input_df.iloc[-6:][['date', 'ddd_demand']].copy()
        last_6['date'] = last_6['date'].dt.strftime('%Y-%m-%d')

        response = {
            'predictions': predictions,
            'best_model': best_model,
            'test_maes': test_maes,
            'last_6_months': last_6.to_dict(orient='records'),
            'model_names': list(model_names)
        }
        logger.info(f"Returning predictions: {len(response['predictions'])} models")
        return response
    except Exception as e:
        logger.error(f"Failed to load predictions: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

@app.get("/api/malaria-maes", summary="Get malaria model MAEs")
async def get_malaria_maes():
    logger.debug("Received request for /api/malaria-maes")
    try:
        mae_path = os.path.join(DATA_DIR, 'test_maes.csv')
        df = pd.read_csv(mae_path)
        if not all(col in df.columns for col in MAE_COLS):
            logger.error(f"Missing columns {MAE_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {MAE_COLS}")
        if (df['mae_sum'] < 0).any():
            logger.error("Negative MAE sum values detected")
            raise ValueError("Negative MAE sum values")

        response = df.to_dict(orient='records')
        logger.info(f"Returning malaria MAEs: {len(response)} rows")
        return response
    except Exception as e:
        logger.error(f"Failed to load malaria MAEs: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load malaria MAEs: {str(e)}")

@app.get("/api/malaria-predictions", summary="Get malaria case predictions")
async def get_malaria_predictions():
    logger.debug("Received request for /api/malaria-predictions")
    try:
        pred_path = os.path.join(DATA_DIR, 'predictions_may_june_july_2025.csv')
        df = pd.read_csv(pred_path)
        if not all(col in df.columns for col in MALARIA_PRED_COLS):
            logger.error(f"Missing columns {MALARIA_PRED_COLS}, got {df.columns.tolist()}")
            raise ValueError(f"Missing columns {MALARIA_PRED_COLS}")

        df['predicted_mal_cases'] = pd.to_numeric(df['predicted_mal_cases'], errors='coerce')
        if df['predicted_mal_cases'].isnull().any():
            logger.error("Invalid predicted_mal_cases values")
            raise ValueError("Invalid predicted_mal_cases values")

        mae_path = os.path.join(DATA_DIR, 'test_maes.csv')
        mae_df = pd.read_csv(mae_path)
        best_model = mae_df.loc[mae_df['mae_sum'].idxmin()]['model_name'] if not mae_df.empty else None
        if not best_model:
            logger.error("No valid model found in test_maes.csv")
            raise ValueError("No valid model found")

        df = df[df['model'] == best_model]
        malaria_data = {}
        districts = df['district'].unique()
        months = ['May', 'June', 'July']
        for district in districts:
            district_data = {'district': district, 'cases': {}, 'status': {}}
            for month in months:
                month_key = f"{month}_2025"
                cases = df[(df['district'] == district) & (df['month'] == month_key)]['predicted_mal_cases']
                case_count = float(cases.iloc[0]) if not cases.empty else 0
                district_data['cases'][month] = case_count
                district_data['status'][month] = 'low' if case_count < 1500 else 'medium' if case_count < 5000 else 'high'
            malaria_data[district] = district_data

        response = {
            'best_model': best_model,
            'data': list(malaria_data.values())
        }
        logger.info(f"Returning malaria predictions: {len(response['data'])} districts")
        return response
    except Exception as e:
        logger.error(f"Failed to load malaria predictions: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load malaria predictions: {str(e)}")

@app.get("/api/historical-malaria", summary="Get historical malaria case data")
async def get_historical_malaria():
    logger.debug("Received request for /api/historical-malaria")
    try:
        historical_path = os.path.join(DATA_DIR, 'malaria_historical.csv')
        if not os.path.exists(historical_path):
            logger.error("Historical malaria data file not found")
            raise FileNotFoundError("Historical malaria data file not found")

        df = pd.read_csv(historical_path)

        if not all(col in df.columns for col in HISTORICAL_COLS):
            logger.error(f"Missing required columns in historical data. Expected: {HISTORICAL_COLS}, Found: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns in historical data")

        df['mal_cases'] = pd.to_numeric(df['mal_cases'], errors='coerce')
        if df['mal_cases'].isnull().any():
            logger.warning("Missing or invalid malaria case values, filling with 0")
            df['mal_cases'].fillna(0, inplace=True)

        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01', errors='coerce')
        if df['date'].isnull().any():
            logger.error("Invalid date formats in historical data")
            raise ValueError("Invalid date formats")

        df = df.sort_values('date')
        last_6_months = df[df['date'] >= df['date'].max() - pd.offsets.MonthBegin(6)].copy()

        response_data = []
        for _, row in last_6_months.iterrows():
            response_data.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'district': row['district'],
                'cases': float(row['mal_cases']),
                'avg_temp_max': float(row.get('avg_temp_max', None)) if 'avg_temp_max' in row else None,
                'avg_temp_min': float(row.get('avg_temp_min', None)) if 'avg_temp_min' in row else None,
                'avg_humidity': float(row.get('avg_humidity', None)) if 'avg_humidity' in row else None,
                'total_precipitation': float(row.get('sum_precipitation', None)) if 'sum_precipitation' in row else None,
                'total_sunshine_hours': float(row.get('sum_sunshine_hours', None)) if 'sum_sunshine_hours' in row else None
            })

        response = {
            'data': response_data
        }

        logger.info(f"Returning historical malaria data: {len(response['data'])} records")
        return response
    except Exception as e:
        logger.error(f"Failed to load historical malaria data: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load historical malaria data: {str(e)}")

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    logger.error(f"404 Not Found: {request.method} {request.url.path} - Client IP: {request.client.host}")
    return JSONResponse(status_code=404, content={"detail": "Resource not found"})