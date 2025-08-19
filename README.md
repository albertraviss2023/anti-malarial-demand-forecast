# Anti-Malarial Demand Forecast

This repository contains a predictive application for anti-malarial demand forecasting, fully containerized using Docker for reliable dependency management and cross-platform deployment. The application consists of two independent Dockerized stacks: the ETL Stack and the Dashboard Applications Stack.

## Prerequisites

Before deploying, ensure the following tools are installed:
- **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/macOS or [Docker CLI/Engine](https://docs.docker.com/engine/install/) for Linux.
- **Git**: Required for cloning the repository.
- **GNU Make** (optional): Simplifies command execution for building and running services.

## Deployment Instructions

### Step 1: Clone the Repository
Clone the repository and switch to the production branch:
```bash
git clone https://github.com/albertraviss2023/anti-malarial-demand-forecast.git
cd anti-malarial-demand-forecast
git checkout main-production
```

### Step 2: Deploy the ETL Stack
The ETL Stack handles scheduled data collection, transformation, and prediction loading. This stack must be deployed first.

1. Navigate to the ETL stack directory:
   ```bash
   cd etl/docker
   ```
2. Build the services:
   ```bash
   make build
   ```
   Alternatively, use:
   ```bash
   docker-compose build --no-cache
   ```
3. Start the services in detached mode:
   ```bash
   make up
   ```
   Alternatively, use:
   ```bash
   docker-compose up -d
   ```

4. Verify the ETL services are running:
   - **Spark UI**: [http://localhost:8081/](http://localhost:8081/)
   - **Airflow UI**: [http://localhost:8080/](http://localhost:8080/)

### Step 3: Deploy the Dashboard Applications Stack
The Dashboard Applications Stack provides visualization services for the forecasting results.

1. From the project root directory, build and start the dashboard services:
   ```bash
   make up
   ```
   Alternatively, use:
   ```bash
   docker-compose -f docker-compose.dashboards.yml build && docker-compose -f docker-compose.dashboards.yml up -d
   ```

2. Access the dashboards:
   - **DDD Dashboard**: [http://localhost:8001/](http://localhost:8001/)
   - **Malaria Dashboard**: [http://localhost:8002/](http://localhost:8002/)

## Troubleshooting Deployment Issues

1. **No Internet Connection**:
   - Ensure an active internet connection, as Docker requires it to pull images from the registry.
2. **Missing Environment Setup**:
   - Verify that Docker, Git, and (optionally) GNU Make are installed and properly configured.
3. **Empty Dashboards**:
   - If dashboards display no data, manually trigger the "insert predictions" DAG in the Airflow UI ([http://localhost:8080/](http://localhost:8080/)).

## Additional Notes
- Ensure the ETL Stack is fully operational before starting the Dashboard Applications Stack, as the dashboards rely on the ETL pipeline for data.
- For further assistance, refer to the repository's documentation or open an issue on GitHub.
