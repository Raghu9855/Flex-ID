# FLEX-ID: Free-Tier Deployment & Operations Guide

This guide provides end-to-end instructions for deploying the **FLEX-ID** backend to 100% free-tier cloud platforms.

---

## 1. Free-Tier Cloud Platform Comparison

| Platform | Free Resources | Cold Start | Docker Support | Best Suited For | Recommended? |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Hugging Face Spaces** | **16 GB RAM**, 2 vCPU, 50 GB Disk | Instant (Persistent) | Native Docker | Machine Learning & Deep Learning serving | **Yes (Top Choice)** |
| **Render.com** | 512 MB RAM, 0.1 vCPU | ~50s after 15 min idle | Native Docker | Lightweight API hosting | **Yes (Secondary)** |
| **Railway.app** | 512 MB RAM, $5 trial credit | Fast | Native Docker | Temporary staging | Alternative |
| **Fly.io** | 256 MB RAM (Free tier) | Instant | Dockerfile | Microservices (Too small for TF) | Not recommended |

> [!TIP]
> **Why Hugging Face Spaces is Recommended:**
> Deep learning libraries (TensorFlow/Keras) have baseline memory requirements of ~280–350 MB. While FLEX-ID runs comfortably in Render's 512 MB container, Hugging Face Spaces grants **16 GB RAM** and **2 vCPUs** for free with zero cold-sleep downtime.

---

## 2. Resource Budgeting & Container Footprint

- **Base Image:** `python:3.11-slim` (multi-stage build)
- **Container Image Size:** ~280 MB (reduced from >1.5 GB by excluding raw training CSVs and client partition dumps via `.dockerignore`)
- **Runtime Memory at Idle:** ~310 MB
- **Runtime Memory Under Load:** ~420 MB
- **Model Parameters:** 52,941 parameters (~206 KB per model weights artifact)
- **GPU Requirement:** None. Dense neural networks execute inference on CPU in under 15ms.

---

## 3. Deployment Walkthrough A: Hugging Face Spaces (Recommended)

Hugging Face Spaces offers free Docker hosting for ML applications with public HTTPS endpoints.

### Step 1: Create a Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click **New Space** (`https://huggingface.co/new-space`).
3. Set:
   - **Space Name:** `flex-id-backend`
   - **License:** `mit` or `apache-2.0`
   - **Space SDK:** Choose **Docker** -> **Blank**.
   - **Space Hardware:** Free (2 vCPU, 16 GB RAM).

### Step 2: Configure Port 7860
Hugging Face Spaces routes traffic through port `7860`.
Create a file named `README.md` at the root of your Hugging Face space repository with this YAML header:

```yaml
---
title: FLEX-ID Intrusion Detection API
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

And in your `Dockerfile` or startup command, set port `7860`:
```dockerfile
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Step 3: Push Repository to Hugging Face
```bash
git remote add space https://huggingface.co/spaces/<your-username>/flex-id-backend
git push --force space main
```

Within 2 minutes, Hugging Face will build the Docker container and display your live API URL at:
`https://<your-username>-flex-id-backend.hf.space/docs`

---

## 4. Deployment Walkthrough B: Render.com

Render provides free Docker container hosting with automated continuous deployment from GitHub.

### Step 1: Push Code to GitHub
Ensure all code and model artifacts (`results/fedproxeachround/round-10-weights.pkl`, `data/scaler.pkl`, `data/label_encoder.pkl`) are committed to your GitHub repository:
```bash
git add .
git commit -m "feat: production backend deployment ready"
git push origin main
```

### Step 2: Create Web Service on Render
1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Fill in the service configuration:
   - **Name:** `flex-id-api`
   - **Region:** Frankfurt (EU) or Oregon (US)
   - **Branch:** `main`
   - **Runtime:** `Docker`
   - **Instance Type:** `Free` (0.1 CPU, 512 MB RAM)

### Step 3: Set Environment Variables on Render
Under **Environment Variables**, configure:

| Key | Value | Notes |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Enables secure error handling |
| `DEBUG` | `false` | Disables verbose debug logging |
| `DATABASE_URL` | `sqlite:///./flex_id.db` | Local persistent SQLite |
| `CORS_ORIGINS` | `*` | Or specify frontend domain |

### Step 4: Deploy and Verify
Click **Create Web Service**. Render will pull the multi-stage Dockerfile, run the build, and deploy.
Your live API will be available at:
`https://flex-id-api.onrender.com/docs`

---

## 5. Deployment Walkthrough C: Local / VPS Docker Compose

To deploy locally or on any cloud VPS (Ubuntu, Debian, EC2) with a managed MySQL 8.0 container:

```bash
# Clone the repository
git clone https://github.com/Raghu9855/Flex-ID.git
cd Flex-ID

# Copy environment template
cp .env.example .env

# Build and start services in background
docker compose up -d --build

# View container logs
docker compose logs -f api
```

The service will be listening on `http://localhost:8000`.

---

## 6. Database Strategy for Free Tier

### Option 1: SQLite on Container Disk (Default & Zero Setup)
- **Setting:** `DATABASE_URL=sqlite:///./flex_id.db`
- **Behavior:** The application automatically runs `Base.metadata.create_all()` and seeds models and benchmark data on startup.
- **Trade-off:** In free-tier cloud containers without persistent volume mounts, the SQLite database is re-initialized if the container restarts. Ideal for portfolio demos and stateless API serving.

### Option 2: Free Cloud Managed Database (Persistent)
If you require persistent audit logging across container restarts:
1. Provision a free PostgreSQL database on [Neon.tech](https://neon.tech/) or free MySQL on [Aiven.io](https://aiven.io/).
2. Set the connection string in your platform's environment variables:
   ```bash
   DATABASE_URL=postgresql://user:password@ep-cool-db.us-east-2.aws.neon.tech/flex_id?sslmode=require
   ```
   or
   ```bash
   DATABASE_URL=mysql+pymysql://avnadmin:password@mysql-project.aivencloud.com:12345/defaultdb
   ```

---

## 7. Post-Deployment Smoke Test (Verification)

Run the following `curl` commands against your deployed URL to verify operational readiness:

### 1. Health & Readiness Check
```bash
curl -X GET "https://<your-deployed-domain>/health"
```
*Expected response:*
```json
{
  "status": "healthy",
  "app_name": "FLEX-ID",
  "version": "1.0.0",
  "environment": "production",
  "database_connected": true,
  "loaded_models_count": 1,
  "default_model_version": "fedprox-v1"
}
```

### 2. Live Flow Classification
```bash
curl -X POST "https://<your-deployed-domain>/api/v1/predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      80.0, 6.0, 150000.0, 25.0, 30.0, 4500.0, 12000.0,
      500.0, 120.0, 800.0, 210.0, 450.0, 110.0, 150.0,
      25000.0, 350.0, 1200.0, 45.0, 50000.0, 20.0, 10.0,
      2.0, 1.0, 0.0, 1.0, 0.0, 512.0, 1024.0, 15.0
    ],
    "model_version": "fedprox-v1"
  }'
```

### 3. Model Explainability Attribution
```bash
curl -X POST "https://<your-deployed-domain>/api/v1/explanations" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      80.0, 6.0, 150000.0, 25.0, 30.0, 4500.0, 12000.0,
      500.0, 120.0, 800.0, 210.0, 450.0, 110.0, 150.0,
      25000.0, 350.0, 1200.0, 45.0, 50000.0, 20.0, 10.0,
      2.0, 1.0, 0.0, 1.0, 0.0, 512.0, 1024.0, 15.0
    ],
    "model_version": "fedprox-v1",
    "top_k": 5
  }'
```

---

## 8. Teardown Instructions

- **Hugging Face:** Navigate to **Settings** -> Scroll to **Delete this Space** -> Confirm.
- **Render:** Open Service -> **Settings** -> Scroll to **Delete Web Service** -> Confirm.
- **Local Docker Compose:**
  ```bash
  docker compose down -v
  ```
