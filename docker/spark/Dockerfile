FROM apache/spark:3.5.0

USER root

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        openjdk-11-jre \
        procps \
        librdkafka-dev \
        freetds-dev \
        libssl-dev \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip3 install \
    pyspark \
    pyiceberg==0.7.1 \
    confluent-kafka==2.5.3 \
    minio==7.2.8

# Add required jars
RUN mkdir -p /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.6.1/iceberg-spark-runtime-3.5_2.12-1.6.1.jar -O /opt/spark/jars/iceberg-spark-runtime.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.6.2/kafka-clients-3.6.2.jar -O /opt/spark/jars/kafka-clients.jar

WORKDIR /opt/spark

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
