FROM apache/nifi:2.0.0-M4

USER root

# Install wget
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Download Microsoft SQL Server JDBC driver
RUN wget -P /opt/nifi/nifi-current/lib/ \
    https://repo1.maven.org/maven2/com/microsoft/sqlserver/mssql-jdbc/12.8.1.jre11/mssql-jdbc-12.8.1.jre11.jar

# Switch back to nifi user
USER nifi
