FROM apache/airflow:2.7.3

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        default-libmysqlclient-dev \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-spark==4.1.0 \
    apache-airflow-providers-postgres==5.7.0 \
    psycopg2-binary==2.9.9 \
    pyspark==3.5.0