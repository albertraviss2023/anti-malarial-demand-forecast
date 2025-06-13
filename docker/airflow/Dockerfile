FROM apache/airflow:2.7.3

USER root

# Install system dependencies including Java and procps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        default-libmysqlclient-dev \
        build-essential \
        openjdk-11-jdk \
        procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for spark-submit compatibility
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

USER airflow

# Add airflow user's pip bin to PATH
ENV PATH=/home/airflow/.local/bin:$PATH

# Install required Python packages
RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-spark==4.1.0 \
    apache-airflow-providers-postgres==5.7.0 \
    psycopg2-binary==2.9.9 \
    pyspark==3.5.0
