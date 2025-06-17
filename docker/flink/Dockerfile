FROM flink:1.18.1-scala_2.12-java11

USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Download Iceberg and Kafka JARs
RUN mkdir -p /opt/flink/lib
RUN wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-flink-runtime-1.18/1.6.1/iceberg-flink-runtime-1.18-1.6.1.jar -O /opt/flink/lib/iceberg-flink-runtime-1.18-1.6.1.jar
RUN wget -q https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.2.0-1.18/flink-sql-connector-kafka-3.2.0-1.18.jar -O /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.18.jar

RUN pip3 install --no-cache-dir pyiceberg==0.7.1 confluent-kafka==2.5.3 minio==7.2.8

USER flink
