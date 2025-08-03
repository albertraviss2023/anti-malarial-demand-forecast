from airflow import DAG
from airflow.providers.papermill.operators.papermill import PapermillOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
}

with DAG(
    dag_id='etl_notebook_pipeline',
    default_args=default_args,
    schedule_interval='0 2 5 * *',  # Run at 2:00 AM on the 5th of every month
    catchup=False
) as dag:

    extract_task = PapermillOperator(
        task_id='extract_data',
        input_nb='/opt/notebooks/extract.ipynb',
        output_nb='/opt/airflow/logs/extract_output_{{ ds }}.ipynb',
        parameters={"cwd": "/opt/notebooks"}
    )

    transform_task = PapermillOperator(
        task_id='transform_data',
        input_nb='/opt/notebooks/Transform.ipynb',
        output_nb='/opt/airflow/logs/transform_output_{{ ds }}.ipynb',
        parameters={"cwd": "/opt/notebooks"}
    )

    load_task = PapermillOperator(
        task_id='load_data',
        input_nb='/opt/notebooks/Load.ipynb',
        output_nb='/opt/airflow/logs/load_output_{{ ds }}.ipynb',
        parameters={"cwd": "/opt/notebooks"}
    )

    extract_task >> transform_task >> load_task
