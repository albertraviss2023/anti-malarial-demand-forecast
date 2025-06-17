FROM jupyter/pyspark-notebook:spark-3.5.0

USER root
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /opt/spark/jars
RUN wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.6.1/iceberg-spark-runtime-3.5_2.12-1.6.1.jar -O /opt/spark/jars/iceberg-spark-runtime-3.5_2.12-1.6.1.jar
RUN wget -q https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.6.2/kafka-clients-3.6.2.jar -O /opt/spark/jars/kafka-clients-3.6.2.jar


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER $NB_UID
