FROM apache/spark:3.5.0
USER root
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install pyspark
WORKDIR /opt/spark