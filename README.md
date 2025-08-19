## ETL Development Branch – Overview

This branch is responsible for running **monthly data collection** using Python-based Airflow DAGs.  
It is a key component for **refreshing monthly data predictions**.  

**Note:** Deployment instructions for the Airflow data refresh service are included in the **production branch**.

---

## Airflow Installation & Environment

- **Installation:** Airflow is installed using a **Docker image**, ensuring a consistent environment for all developers and during production deployment.  
- **Dependencies:** All libraries and packages required for data collection, transformation, and loading are pre-installed in the Docker image, including:
  - Python libraries: `pandas`, `numpy`, `sqlalchemy`, `psycopg2`, etc.
  - Airflow providers and operators relevant for database, file, and Python task execution.
- **Configuration:** The Docker image contains the correct **Airflow version**, scheduler, webserver, and necessary environment variables for DAG execution.

**Advantages of using Docker for Airflow:**
1. **Environment Consistency:** Ensures all developers and environments use the exact same Airflow setup.
2. **Easy Setup:** No need to manually install Airflow and dependencies on local machines.
3. **Isolation:** Keeps Airflow and its dependencies separate from other system packages.
4. **Portability:** The Docker image can be deployed to any server or cloud environment without modification.
5. **Reproducibility:** DAGs and workflows behave the same way across development, staging, and production environments.

---

## Key Concepts for Users

Below are the major highlights and concepts you need to understand about Airflow and pipelines in this ETL branch:

1. **DAG (Directed Acyclic Graph):**  
   - Represents the workflow of tasks and their dependencies.  
   - Each DAG corresponds to a monthly data refresh workflow in this branch.

2. **Tasks & Operators:**  
   - Tasks are the individual steps in a DAG.  
   - Python-based ETL steps use `PythonOperator` to run Python functions.

3. **Scheduling:**  
   - DAGs are scheduled to run **monthly**, ensuring data is collected and processed regularly.  
   - Schedule intervals are defined in the DAG file using `schedule_interval`.

4. **Dependencies:**  
   - Task dependencies define the **execution order**.  
   - Airflow ensures that downstream tasks only run after upstream tasks have successfully completed.

5. **XComs (Cross-communications):**  
   - Used to **pass data between tasks** in a DAG.  
   - Useful for sharing results from one step (e.g., data extraction) to another (e.g., data transformation).

6. **Logging & Monitoring:**  
   - Airflow provides detailed logs for each task.  
   - Users can monitor DAG runs, successes, and failures via the Airflow UI.

7. **Idempotency:**  
   - Each DAG and task should be **repeatable** without causing duplication or data corruption.  
   - Critical for monthly ETL pipelines where tasks may be retried automatically.

8. **Version Control:**  
   - DAGs in this branch are **development versions**.  
   - Production deployment, including Airflow setup, credentials, and scheduler configuration, is maintained in the **production branch**.

9. **Error Handling & Retries:**  
   - Tasks can be configured with **retry policies**.  
   - Failed tasks are retried according to the DAG configuration to ensure robustness.

10. **Environment & Dependencies:**  
    - DAGs rely on the **Python environment** and installed libraries.  
    - Using the Docker image ensures all required packages are pre-installed, avoiding missing dependency issues during execution.

---

This overview provides developers with a clear understanding of **how the ETL branch functions**, the **Airflow setup using Docker**, and the **key concepts needed to maintain or extend monthly data collection pipelines**.
