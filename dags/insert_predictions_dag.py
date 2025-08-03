from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow/scripts')

from ingest_predictions import parse_and_insert  # this is your Python script logic

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 8, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='insert_predictions_dag',
    schedule_interval='0 9 8 * *',  # Every 8th day at 9AM
    default_args=default_args,
    catchup=False,
    description='Insert malaria and DDD predictions into PostgreSQL',
) as dag:

    insert_predictions = PythonOperator(
        task_id='parse_and_insert_predictions',
        python_callable=parse_and_insert
    )

    insert_predictions
