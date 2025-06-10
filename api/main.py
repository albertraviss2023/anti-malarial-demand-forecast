from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
import pandas as pd
import os
from datetime import datetime
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Malaria Dashboard API",
    description="API for malaria case tracking and forecasting"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Helper functions
def get_district_coordinates(district: str) -> Dict[str, float]:
    """Return coordinates for a given district"""
    districts_coords = {
        "Kampala": {"latitude": 0.3136, "longitude": 32.5811},
        "Gulu": {"latitude": 2.7666, "longitude": 32.3057},
        # Add all other districts here
    }
    return districts_coords.get(district)

# API Endpoints
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/historical-malaria")
async def get_historical_malaria() -> List[Dict]:
    """Get historical malaria case data"""
    try:
        file_path = os.path.join(DATA_DIR, 'historical', 'malaria_historical.csv')
        df = pd.read_csv(file_path)
        
        # Process and validate data
        required_cols = ['year', 'month', 'district', 'mal_cases']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in historical data")
        
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values('date')
        
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/malaria-predictions")
async def get_malaria_predictions() -> Dict:
    """Get malaria predictions for May-July 2025"""
    try:
        file_path = os.path.join(DATA_DIR, 'predictions_may_june_july_2025.csv')
        df = pd.read_csv(file_path)
        
        # Process predictions
        results = []
        districts = df['district'].unique()
        
        for district in districts:
            district_data = df[df['district'] == district]
            entry = {
                'district': district,
                'cases': {},
                'status': {},
                'coordinates': get_district_coordinates(district)
            }
            
            for _, row in district_data.iterrows():
                month = row['month'].split()[0]  # Extract month name
                cases = float(row['predicted_mal_cases'])
                entry['cases'][month] = cases
                
                # Determine status based on cases
                if cases > 1000:
                    status = 'High'
                elif cases > 500:
                    status = 'Medium'
                else:
                    status = 'Low'
                entry['status'][month] = status
            
            results.append(entry)
        
        return JSONResponse(content={'data': results})
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    