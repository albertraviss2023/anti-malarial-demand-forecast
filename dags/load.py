from airflow import DAG
from airflow.providers.papermill.operators.papermill import PapermillOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
}

with DAG(
    dag_id='load_notebook_only_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    load_task = PapermillOperator(
        task_id='load_data',
        input_nb='/opt/notebooks/Load.ipynb',
        output_nb='/opt/airflow/logs/load_output_{{ ds }}.ipynb',
        parameters={"cwd": "/opt/notebooks"}
    )
