import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Uganda Malaria Dashboard API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/api/districts")
async def get_districts():
    try:
        df = pd.read_csv("data/districts.csv")
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/malaria_predictions")
async def get_malaria_predictions():
    try:
        maes_df = pd.read_csv("data/test_maes.csv")
        best_model = maes_df.loc[maes_df['mae_sum'].idxmin()]['model_name']

        df = pd.read_csv("data/malaria_predictions.csv")
        df = df[df['model'] == best_model]
        pivoted = df.pivot(index="district", columns="month", values="predicted_mal_cases").reset_index()
        pivoted.columns = ["district", "May", "June", "August"]
        result = pivoted.to_dict(orient="records")
        threshold = 10000
        for district in result:
            district["cases"] = {
                "May": district["May"],
                "June": district["June"],
                "August": district["August"]
            }
            district["status"] = {
                "May": "high" if district["May"] > threshold else "medium" if district["May"] > threshold * 0.7 else "low",
                "June": "high" if district["June"] > threshold else "medium" if district["June"] > threshold * 0.7 else "low",
                "August": "high" if district["August"] > threshold else "medium" if district["August"] > threshold * 0.7 else "low"
            }
            for month in ["May", "June", "August"]:
                del district[month]
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/model_metrics")
async def get_model_metrics():
    try:
        df = pd.read_csv("data/test_maes.csv")
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/malaria_historical")
async def get_malaria_historical():
    try:
        df = pd.read_csv("data/malaria_historical.csv")
        
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m')
        
        malaria_agg = df.groupby(['year', 'month', 'date'])['mal_cases'].sum().reset_index()
        malaria_data = malaria_agg[['date', 'mal_cases']].rename(columns={'mal_cases': 'cases'}).to_dict(orient="records")
        
        climate_agg = df.groupby(['year', 'month', 'date']).agg({
            'avg_temp_max': 'mean',
            'avg_temp_min': 'mean',
            'avg_humidity': 'mean',
            'sum_precipitation': 'sum',
            'sum_sunshine_hours': 'sum'
        }).reset_index()
        climate_data = climate_agg.to_dict(orient="records")
        
        corr_data = df.groupby(['year', 'month']).agg({
            'mal_cases': 'sum',
            'avg_temp_max': 'mean',
            'avg_temp_min': 'mean',
            'avg_humidity': 'mean',
            'sum_precipitation': 'sum',
            'sum_sunshine_hours': 'sum'
        }).reset_index()
        
        correlation_coeffs = {
            'avg_temp_max': corr_data['mal_cases'].corr(corr_data['avg_temp_max']),
            'avg_temp_min': corr_data['mal_cases'].corr(corr_data['avg_temp_min']),
            'avg_humidity': corr_data['mal_cases'].corr(corr_data['avg_humidity']),
            'sum_precipitation': corr_data['mal_cases'].corr(corr_data['sum_precipitation']),
            'sum_sunshine_hours': corr_data['mal_cases'].corr(corr_data['sum_sunshine_hours'])
        }
        
        corr_data = corr_data.to_dict(orient="records")
        
        return {
            "malaria_cases": malaria_data,
            "climate": climate_data,
            "correlations": corr_data,
            "correlation_coeffs": correlation_coeffs
        }
    except Exception as e:
        return {"error": str(e)}