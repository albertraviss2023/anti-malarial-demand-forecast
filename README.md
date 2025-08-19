# District Malaria CASES Dashboard – Development Branch

This repository contains the **Development branch** for the District Malaria CASES Dashboard, which has also since been merged into the **production branch**.  
The dashboard can still be run locally for testing or development purposes.

---

## 🚀 Features

- Interactive dashboard for tracking malaria cases.
- Built with **FastAPI** and **Uvicorn**.
- PostgreSQL backend for storing dashboard data.
- Easy local setup with Docker for PostgreSQL.

---

## 🛠️ Setup & Launch Instructions

1. **Start the PostgreSQL Docker service** (required for the dashboard):
   ```bash
   docker-compose up -d postgres

2. **Launch FastAPI server** :
   ```bash
   uvicorn app.main:app --reload

3. **Access UI** :
   via link : http://127.0.0.1:8000)
