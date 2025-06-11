from datetime import datetime
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

with DAG(
    dag_id='spark_job_dag',
    start_date=datetime(2025, 6, 10),
    schedule_interval=None,
) as dag:
    spark_job = SparkSubmitOperator(
        task_id='run_spark_job',
        application='/opt/spark_jobs/job.py',
        conn_id='spark_default',
        executor_memory='1g',
        total_executor_cores=1,
    )
    