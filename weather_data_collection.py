#!/usr/bin/env python3
"""
Uganda District Weather Data Collector

Usage:
1. First install dependencies: pip install -r requirements.txt
2. Then run: python weather_data_collector.py

Data will be saved in: data/weather/
"""

import os
import json
import requests
import pandas as pd
from time import sleep
import random
from datetime import datetime
from dotenv import load_dotenv

# Configuration
load_dotenv()  # Load environment variables from .env file
DISTRICTS_JSON_PATH = "data/districts.json"
WEATHER_DATA_FOLDER = "data/weather"
START_YEAR = 2017
END_YEAR = 2025
API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RETRIES = 5
REQUEST_TIMEOUT = 30  # seconds

def check_dependencies():
    """Verify all required packages are installed"""
    try:
        import requests
        import pandas
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}")
        print("Please install all requirements with: pip install -r requirements.txt")
        exit(1)

def setup_data_directory():
    """Ensure data directory structure exists"""
    os.makedirs(WEATHER_DATA_FOLDER, exist_ok=True)
    if not os.path.exists(DISTRICTS_JSON_PATH):
        print(f"Error: Districts JSON file not found at {DISTRICTS_JSON_PATH}")
        exit(1)

def load_districts():
    """Load districts data from JSON file"""
    with open(DISTRICTS_JSON_PATH, "r") as f:
        districts = json.load(f)
    print(f"Loaded {len(districts)} districts from {DISTRICTS_JSON_PATH}")
    return pd.DataFrame(districts)

def fetch_weather_data(latitude, longitude, start_date, end_date):
    """Fetch weather data with retry logic"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,sunshine_duration",
        "timezone": "Africa/Kampala"
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5 * attempt))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                sleep(retry_after)
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {str(e)}")
            if attempt < MAX_RETRIES:
                sleep(5 * attempt)  # Exponential backoff
                
    print(f"Max retries reached for {latitude},{longitude}")
    return None

def process_district_data(district_name, lat, lon, year):
    """Process and save data for one district-year"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-03-31" if year == 2025 else f"{year}-12-31"
    
    print(f"Fetching {year} data for {district_name}...")
    data = fetch_weather_data(lat, lon, start_date, end_date)
    
    if not data or "daily" not in data:
        print(f"❌ No data for {district_name} in {year}")
        return False
        
    df = pd.DataFrame(data["daily"])
    df["district"] = district_name
    df["latitude"] = lat
    df["longitude"] = lon
    df["time"] = pd.to_datetime(df["time"])
    
    filename = f"{WEATHER_DATA_FOLDER}/{district_name}_{year}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename} ({len(df)} records)")
    return True

def main():
    print("\nUganda District Weather Data Collector")
    print("=" * 40)
    
    # Verify environment
    check_dependencies()
    setup_data_directory()
    
    # Load districts
    districts = load_districts()
    total_districts = len(districts)
    
    # Process each district
    for idx, row in districts.iterrows():
        district_name = row["district"].replace(" ", "_")
        lat, lon = row["latitude"], row["longitude"]
        
        print(f"\n[{idx+1}/{total_districts}] Processing {district_name}...")
        
        for year in range(START_YEAR, END_YEAR + 1):
            success = process_district_data(district_name, lat, lon, year)
            
            # Add delay between years
            if year < END_YEAR and success:
                sleep(random.uniform(2, 5))
                
            if not success:
                break
                
        # Add delay between districts
        if idx < total_districts - 1:
            sleep(random.uniform(5, 10))
    
    print("\n" + "=" * 40)
    print("Data collection complete!")
    print(f"Weather data saved to: {WEATHER_DATA_FOLDER}")

if __name__ == "__main__":
    main()