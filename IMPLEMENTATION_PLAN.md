# FLEX-ID Backend-First Transformation: Implementation Plan

**Author:** Senior Backend Engineer & AI Systems Architect  
**Project:** FLEX-ID (Federated Learning Explainable Intrusion Detection System)  
**Target Roles:** Software Developer, Backend Engineer, Python Developer, AI Systems Engineer  
**Status:** Phase 1 Complete (Plan Ready for Execution)

---

## 1. Target Architecture & System Design

To transform FLEX-ID from an academic prototype with an insecure Node.js wrapper into a production-grade, API-driven AI system, we will implement a clean, layered architecture powered by **FastAPI**, **SQLAlchemy ORM**, **Pydantic v2**, and a modular **Service-Repository** pattern.

```
                                  ┌────────────────────────┐
                                  │   Clients / Web UI     │
                                  │ (React / Curl / Admin) │
                                  └───────────┬────────────┘
                                              │ HTTP / JSON
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND APPLICATION                               │
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                  API ROUTERS                                      │  │
│  │   /health   /api/v1/models   /api/v1/predictions   /api/v1/experiments   /xai     │  │
│  └─────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                            │ Dependency Injection (DB, Services)        │
│                                            ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                 SERVICE LAYER                                     │  │
│  │   ┌────────────────────┐ ┌────────────────────┐ ┌──────────────────────────────┐  │  │
│  │   │  InferenceService  │ │ ExplanationService │ │ FederatedExperimentService   │  │  │
│  │   │  (Model Manager,   │ │ (SHAP Kernel /     │ │ (Background Task Runner,     │  │  │
│  │   │   Scaler Pipeline) │ │  Fast Attribution) │ │  Flower Coordinator)         │  │  │
│  │   └─────────┬──────────┘ └──────────┬─────────┘ └──────────────┬───────────────┘  │  │
│  └─────────────┼───────────────────────┼──────────────────────────┼──────────────────┘  │
│                │                       │                          │                     │
│                ▼                       ▼                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                             REPOSITORY & DATA ACCESS                              │  │
│  │       PredictionRepository       ModelRepository       ExperimentRepository       │  │
│  └─────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                            │ SQLAlchemy Sessions                        │
│                                            ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                               PERSISTENCE & STORAGE                               │  │
│  │        SQLite (Dev / Test) / MySQL (Prod)  │  Artifact Store (Pickles / PNGs)     │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

We will adopt the following modular structure in the project root:

```text
flex-id/
├── app/
│   ├── main.py                     # FastAPI application factory, lifespan, CORS, middleware
│   ├── api/
│   │   ├── dependencies.py         # DB session, auth, service dependencies
│   │   └── v1/
│   │       ├── router.py           # Master v1 router aggregating all endpoints
│   │       ├── health.py           # Health checks and system readiness probe
│   │       ├── models.py           # Model registry and metadata inspection
│   │       ├── predictions.py      # Single and batch network traffic inference
│   │       ├── experiments.py      # FL experiment creation, status, and metrics
│   │       └── explanations.py     # Feature attribution and SHAP explanations
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (ENV variables, DB URL, paths)
│   │   ├── logging.py              # Structured logging configuration
│   │   └── exceptions.py           # Global HTTP exception handlers
│   ├── db/
│   │   ├── session.py              # SQLAlchemy engine and sessionmaker
│   │   └── base.py                 # Declarative Base
│   ├── models/                     # SQLAlchemy ORM models
│   │   ├── experiment.py           # Experiment configuration and status
│   │   ├── experiment_metric.py    # Per-round metrics (loss, acc, f1)
│   │   ├── model_record.py         # Registered model checkpoints and metadata
│   │   └── prediction_log.py       # Audited prediction requests and latency
│   ├── schemas/                    # Pydantic v2 validation models
│   │   ├── health.py               # Health response schema
│   │   ├── prediction.py           # TrafficFeatureInput, PredictionResponse, BatchPrediction
│   │   ├── model.py                # ModelMetadataResponse
│   │   ├── experiment.py           # ExperimentCreate, ExperimentResponse, MetricRecord
│   │   └── explanation.py          # ExplanationRequest, FeatureAttributionResponse
│   ├── repositories/               # Data access layer
│   │   ├── experiment_repository.py
│   │   ├── model_repository.py
│   │   └── prediction_repository.py
│   └── services/                   # Business and ML logic
│       ├── model_manager.py        # Safe Keras model loading & weights caching
│       ├── inference_service.py    # Preprocessing scaling + prediction pipeline
│       ├── explanation_service.py  # Fast SHAP and local feature importance
│       └── fl_service.py           # Asynchronous FL training execution & tracking
├── ml/                             # Existing federated learning engine (preserved)
│   ├── model.py                    # DNN architecture
│   ├── aggregation.py              # Robust aggregators (Krum, Median, Trimmed Mean)
│   ├── 1_process_data.py           # Preprocessing script (enhanced with scaler export)
│   ├── 2_create_partitions.py      # Non-IID Dirichlet split
│   ├── 4_server.py                 # Flower FL server
│   ├── client.py                   # Honest FL client
│   ├── client_attack.py            # Adversarial FL client
│   └── explain_model.py            # Federated SHAP module
├── tests/                          # Automated Pytest suite
│   ├── conftest.py                 # Fixtures (TestClient, in-memory DB, sample features)
│   ├── test_health.py              # Health check tests
│   ├── test_predictions.py         # Single & batch inference tests
│   ├── test_models.py              # Model retrieval tests
│   ├── test_experiments.py         # Experiment creation and metrics tests
│   ├── test_explanations.py        # SHAP & attribution endpoint tests
│   └── test_inference_service.py   # Preprocessing and scaler validation
├── Dockerfile                      # Production multi-stage Docker container
├── docker-compose.yml              # API + MySQL + Persistent Volume orchestration
├── requirements.txt                # Consolidated, reproducible dependencies
├── .env.example                    # Sample environment variables
├── PROJECT_AUDIT.md                # Completed Phase 1 Audit
└── IMPLEMENTATION_PLAN.md          # This roadmap
```

---

## 3. Database Schema Design (SQLAlchemy ORM)

### 3.1 `experiments` Table
* `id` (String / UUID, PK): Unique experiment identifier.
* `name` (String): Human-readable name (e.g., `fedprox_label_flip_exp_01`).
* `strategy` (String): `fedavg`, `fedprox`, `krum`, `multikrum`, `trimmed_mean`, `median`.
* `rounds` (Integer): Total configured rounds (e.g., 10 or 30).
* `num_clients` (Integer): Number of participating clients (4, 8, 16).
* `proximal_mu` (Float): Regularization coefficient for FedProx.
* `is_attack` (Boolean): True if adversarial client is included.
* `attack_type` (String, nullable): `flip`, `noise`, `backdoor`, `byzantine`, `adaptive`.
* `status` (String): `pending`, `running`, `completed`, `failed`.
* `error_message` (Text, nullable): Capture stack trace if failed.
* `created_at` (DateTime): UTC timestamp.
* `completed_at` (DateTime, nullable): UTC timestamp.

### 3.2 `experiment_metrics` Table
* `id` (Integer, PK, auto-increment).
* `experiment_id` (String, FK -> `experiments.id`, indexed).
* `round_number` (Integer).
* `train_loss` (Float, nullable).
* `accuracy` (Float, nullable).
* `f1_score` (Float, nullable).
* `created_at` (DateTime).

### 3.3 `model_records` Table
* `id` (String / UUID, PK): Unique model record ID.
* `model_version` (String, indexed): e.g., `fedprox-v1`, `fedavg-v1`.
* `strategy` (String): Aggregation algorithm.
* `weights_path` (String): Path to weights file on disk.
* `input_features_count` (Integer): Default 29.
* `num_classes` (Integer): Default 13 (or 2 for binary).
* `accuracy` (Float, nullable): Evaluated test accuracy.
* `macro_f1` (Float, nullable): Evaluated macro F1.
* `is_active` (Boolean): True if this model is default for live inference.
* `created_at` (DateTime).

### 3.4 `prediction_logs` Table
* `id` (String / UUID, PK).
* `model_id` (String, FK -> `model_records.id`, indexed).
* `features_hash` (String, nullable): SHA256 of feature payload for deduplication.
* `predicted_class` (String): e.g., `Benign`, `DoS attacks-Hulk`, `DDOS attack-HOIC`.
* `predicted_class_id` (Integer): Numeric label.
* `confidence` (Float): Probability from Softmax (0.0 - 1.0).
* `latency_ms` (Float): Execution latency in milliseconds.
* `created_at` (DateTime, indexed).

---

## 4. Phase-by-Phase Execution Roadmap

### Phase 2: Core Foundation & Configuration
* Create `app/core/config.py` using `pydantic-settings`.
* Create `app/core/logging.py` for structured log output.
* Create `app/core/exceptions.py` with custom exceptions and exception handlers.
* Build database connection engine in `app/db/session.py` supporting SQLite for development/testing and MySQL for production.
* Implement database models in `app/models/`.

### Phase 3: Machine Learning Infrastructure & Scaler Export
* **Fix the Preprocessing Scaler Gap:**
  * Extract or recompute the `MinMaxScaler` on `data/processed_data.csv` features and save as `data/scaler.pkl`.
  * Ensure the exact 29 feature columns and order are strictly defined and validated.
* **Build Safe Model Manager (`app/services/model_manager.py`):**
  * Load DNN architecture from `ml/model.py`.
  * Restore weights from `results/fedproxeachround/round-10-weights.pkl` or `results/fedavgeachround/round-10-weights.pkl`.
  * Maintain an in-memory cache to prevent disk reads on every HTTP request.
* **Build Inference Service (`app/services/inference_service.py`):**
  * Accept feature dictionary or array.
  * Validate against schema.
  * Apply `scaler.transform()`.
  * Run `model.predict()`, calculate softmax probabilities, extract class name via `label_encoder.pkl`.
  * Support efficient batch inference with bounded memory usage.

### Phase 4: Explainability Service (SHAP Integration)
* **Build `app/services/explanation_service.py`:**
  * Load precomputed global feature importance from `results/shap_summary.json` for instant summary queries.
  * Implement lightweight local feature attribution (fast perturbation or low-sample KernelSHAP) for single predictions with strict timeout protection.
  * Return top-$K$ positive and negative contributing features for the flagged intrusion.

### Phase 5: Federated Learning Backend Service
* **Build `app/services/fl_service.py`:**
  * Define experiment submission and validation.
  * Implement background execution using Python's `asyncio` / `BackgroundTasks` (or daemon threads) without blocking the HTTP server.
  * Record round-by-round training loss and evaluation metrics into the database.
  * Seed database with existing 10-round historical results (`FedAvg`, `FedProx`, `Attack` runs) so initial queries immediately return rich operational metrics.

### Phase 6: REST API Layer Implementation
* Build Pydantic schemas in `app/schemas/`.
* Implement API routes:
  * `GET /health` — Check DB connectivity and model readiness.
  * `GET /api/v1/models` — List registered model versions.
  * `GET /api/v1/models/{model_id}` — Get model details, metrics, and weights info.
  * `POST /api/v1/predictions` — Single traffic flow prediction with confidence.
  * `POST /api/v1/predictions/batch` — Batch traffic prediction with concurrency control.
  * `GET /api/v1/predictions/{prediction_id}` — Retrieve audited prediction record.
  * `POST /api/v1/experiments` — Trigger or register a federated experiment.
  * `GET /api/v1/experiments` — List federated experiments and their status.
  * `GET /api/v1/experiments/{experiment_id}` — Get experiment details.
  * `GET /api/v1/experiments/{experiment_id}/metrics` — Get per-round learning curves.
  * `POST /api/v1/explanations` — Compute or retrieve feature attributions for a packet flow.
* Add OpenAPI documentation metadata (`title`, `version`, `tags`).

### Phase 7: Security, Input Validation & Reliability
* Configure CORS middleware.
* Enforce payload limits (e.g., max 1000 records in batch inference).
* Standardize error responses (400 Bad Request, 404 Not Found, 422 Validation Error, 500 Internal Server Error) without leaking internal tracebacks.
* Add request ID and latency logging headers.

### Phase 8: Automated Testing Suite
* Build comprehensive tests in `tests/`:
  * Health check tests (`test_health.py`).
  * Valid/invalid input validation tests for predictions (`test_predictions.py`).
  * Batch prediction limits and performance tests.
  * Preprocessing and feature alignment tests (`test_inference_service.py`).
  * Model retrieval and metadata tests (`test_models.py`).
  * Experiment creation and status tracking tests (`test_experiments.py`).
  * SHAP explanation tests (`test_explanations.py`).
* Execute pytest and report exact pass/fail counts.

### Phase 9: Containerization & Docker Setup
* Create a modern, multi-stage `Dockerfile`:
  * Use official `python:3.11-slim` or `3.13-slim`.
  * Install dependencies with pip cache management.
  * Create non-root system user (`appuser`).
  * Add `HEALTHCHECK` pointing to `/health`.
* Create `docker-compose.yml`:
  * Service 1: `api` (FastAPI backend).
  * Service 2: `db` (MySQL 8.0) with persistent volume.
  * Health dependency (`depends_on` condition `service_healthy`).
  * Dedicated bridge network.
* Update `.env.example` and setup instructions.

### Phase 10: Documentation & Resume Engineering
* Rewrite `README.md` to highlight:
  * Backend AI Architecture & API endpoints.
  * Honest benchmark metrics (Accuracy, Macro F1, MCC).
  * Docker execution steps.
  * Sample curl requests/responses.
* Draft 3 to 5 targeted, ATS-optimized resume bullet points for Software Developer, Backend Engineer, and AI Systems roles.

---

## 5. Verification & Safety Guarantees

* **Preserve Existing Artifacts:** Existing files in `ml/`, `results/`, `data/`, and `frontend/` will not be deleted or overwritten.
* **No Mocking in Production:** The actual Keras DNN model weights (`round-10-weights.pkl`) and label encoder will power the live inference service.
* **Empirical Integrity:** Results reported in documentation and resume will strictly derive from verified metrics in `results/comparison_results_*.json` and `results/metrics_summary.json`.
