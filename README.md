# National Anti-Malarial DDD – Dashboard

This repository contains the **Development branch** for the National Anti-Malarial DDD dashboard, which has since been merged into the **production branch**.  
The dashboard can still be run locally for testing or development purposes.

---

## 🚀 Features

- Interactive dashboard for tracking anti-malarial demand.
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

