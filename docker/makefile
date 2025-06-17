# Makefile for managing multi-compose Docker stacks with dependencies

# Compose files for individual services
COMPOSE_CORE=docker-compose.yml
COMPOSE_MINIO=docker-compose-minio.yml
COMPOSE_KAFKA=docker-compose-kafka.yml
COMPOSE_ICEBERG=docker-compose-iceberg.yml
COMPOSE_JUPYTER=docker-compose-jupyter.yml
COMPOSE_NIFI=docker-compose-nifi.yml
COMPOSE_MSSQL=docker-compose-mssql.yml
COMPOSE_TRINO=docker-compose-trino.yml

# Base docker-compose command helper with files
dc = docker-compose

# Compose files joined for convenience, separated by space
# Compose files to bring up core dependencies for a Jupyter stack
JUPYTER_STACK = -f $(COMPOSE_CORE) -f $(COMPOSE_MINIO) -f $(COMPOSE_JUPYTER) -f $(COMPOSE_ICEBERG)
NIFI_STACK    = -f $(COMPOSE_MINIO) -f $(COMPOSE_KAFKA) -f $(COMPOSE_MSSQL) -f $(COMPOSE_NIFI)
TRINO_STACK   = -f $(COMPOSE_MINIO) -f $(COMPOSE_ICEBERG) -f $(COMPOSE_TRINO)

.PHONY: help all start-core start-minio start-kafka start-iceberg start-jupyter start-nifi start-trino \
        stop-core stop-minio stop-kafka stop-iceberg stop-jupyter stop-nifi stop-trino \
        down clean

help:
	@echo "Usage:"
	@echo "  make start-core       # Start core services (spark-master)"
	@echo "  make start-minio      # Start MinIO"
	@echo "  make start-kafka      # Start Kafka and Zookeeper"
	@echo "  make start-iceberg    # Start Iceberg REST (depends on MinIO)"
	@echo "  make start-jupyter    # Start Jupyter stack with dependencies"
	@echo "  make start-nifi       # Start NiFi stack"
	@echo "  make start-trino      # Start Trino stack"
	@echo "  make stop-<service>   # Stop specific services"
	@echo "  make down             # Stop and remove all containers"
	@echo "  make clean            # Remove all containers, volumes, and networks"

# Start core (spark-master) only
start-core:
	$(dc) -f $(COMPOSE_CORE) up -d --build

start-minio:
	$(dc) -f $(COMPOSE_MINIO) up -d --build minio

start-kafka:
	$(dc) -f $(COMPOSE_KAFKA) up -d --build zookeeper kafka

start-iceberg:
	$(dc) -f $(COMPOSE_MINIO) -f $(COMPOSE_ICEBERG) up -d --build iceberg-rest

start-jupyter:
	$(dc) $(JUPYTER_STACK) up -d --build spark-master minio jupyter iceberg-rest

start-nifi:
	$(dc) $(NIFI_STACK) up -d --build minio kafka mssql nifi

start-trino:
	$(dc) $(TRINO_STACK) up -d --build minio iceberg-rest trino

# Stop specific service(s)
stop-core:
	$(dc) -f $(COMPOSE_CORE) stop 

stop-minio:
	$(dc) -f $(COMPOSE_MINIO) stop minio

stop-kafka:
	$(dc) -f $(COMPOSE_KAFKA) stop zookeeper kafka

stop-iceberg:
	$(dc) -f $(COMPOSE_ICEBERG) stop iceberg-rest

stop-jupyter:
	$(dc) $(JUPYTER_STACK) stop spark-master minio kafka jupyter

stop-nifi:
	$(dc) $(NIFI_STACK) stop minio kafka mssql nifi

stop-trino:
	$(dc) $(TRINO_STACK) stop minio iceberg-rest trino

# Down / clean all services and volumes/networks
down:
	$(dc) -f $(COMPOSE_CORE) -f $(COMPOSE_MINIO) -f $(COMPOSE_KAFKA) -f $(COMPOSE_ICEBERG) \
		-f $(COMPOSE_JUPYTER) -f $(COMPOSE_NIFI) -f $(COMPOSE_MSSQL) -f $(COMPOSE_TRINO) down

clean: down
	docker system prune -f
	docker volume prune -f
	docker network prune -f
