FROM apache/spark:3.5.0
USER root
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y openjdk-11-jre procps && apt-get clean
RUN pip3 install pyspark
WORKDIR /opt/spark

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
