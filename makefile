# Makefile for managing dashboard stack
COMPOSE_FILE=docker-compose.dashboards.yml

up:
	docker-compose -f $(COMPOSE_FILE) up -d

build:
	docker-compose -f $(COMPOSE_FILE) build

rebuild:
	docker-compose -f $(COMPOSE_FILE) build --no-cache

down:
	docker-compose -f $(COMPOSE_FILE) down

logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

restart:
	docker-compose -f $(COMPOSE_FILE) down && docker-compose -f $(COMPOSE_FILE) up -d

status:
	docker-compose -f $(COMPOSE_FILE) ps

exec-ddd:
	docker-compose -f $(COMPOSE_FILE) exec dashboard-ddd sh

exec-malaria:
	docker-compose -f $(COMPOSE_FILE) exec dashboard-malaria sh
