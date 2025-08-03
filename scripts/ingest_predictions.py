import os
import glob
import re
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

# Maps month names to numbers for parsing filenames
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def infer_generated_on(filename, fallback_date):
    # Updated regex to match e.g. June2025 or June_2025
    match = re.search(r'([A-Za-z]+)[\-_]?(\d{4})', filename)
    if match:
        month_str, year_str = match.groups()
        month_str = month_str.capitalize()
        if month_str in MONTH_MAP:
            generated_date = datetime(int(year_str), MONTH_MAP[month_str], 7).date()
            print(f"Inferred generated_on={generated_date} from filename={filename}")
            return generated_date
    print(f"Using fallback generated_on={fallback_date} for filename={filename}")
    return fallback_date

def create_tables_if_not_exist(cur):
    # Table for national DDD predictions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ddd_predictions (
            model TEXT NOT NULL,
            date DATE NOT NULL,
            predicted_demand FLOAT NOT NULL,
            generated_on DATE NOT NULL,
            PRIMARY KEY (model, date, generated_on)
        );
    """)

    # Table for malaria case predictions per district
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

def parse_and_insert():
    fallback_generated_on = (datetime.today() - timedelta(days=1)).date()
    results_dir = "/opt/airflow/results"

    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "prod_db"),
        user=os.getenv("POSTGRES_USER", "airflow"),
        password=os.getenv("POSTGRES_PASSWORD", "airflow"),
        host="postgres",
        port="5432"
    )
    cur = conn.cursor()

    print("Creating tables if not exist...")
    create_tables_if_not_exist(cur)

    # Insert DDD predictions
    ddd_files = glob.glob(os.path.join(results_dir, "ddd_predictions_*.csv"))
    print(f"Found DDD prediction files: {ddd_files}")
    for file in ddd_files:
        print(f"Processing {file}")
        df = pd.read_csv(file)
        generated_on = infer_generated_on(os.path.basename(file), fallback_generated_on)
        df['generated_on'] = generated_on
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO ddd_predictions (model, date, predicted_demand, generated_on)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model, date, generated_on) DO NOTHING;
            """, (row['model_name'], row['date'], row['predicted_demand'], generated_on))

    # Insert malaria case predictions
    malaria_files = glob.glob(os.path.join(results_dir, "predictions_*.csv"))
    print(f"Found malaria prediction files: {malaria_files}")
    for file in malaria_files:
        print(f"Processing {file}")
        df = pd.read_csv(file)
        generated_on = infer_generated_on(os.path.basename(file), fallback_generated_on)
        df['generated_on'] = generated_on
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO malaria_predictions (district, model, month, predicted_mal_cases, generated_on)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (district, model, month, generated_on) DO NOTHING;
            """, (row['district'], row['model'], row['month'], row['predicted_mal_cases'], generated_on))

    conn.commit()
    cur.close()
    conn.close()
    print("Data insertion complete.")

if __name__ == "__main__":
    parse_and_insert()
