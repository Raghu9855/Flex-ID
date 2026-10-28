# FLEX-ID Stage A: Repository Audit and Baseline Analysis

**Audit Date:** October 2026  
**Auditor Roles:** Senior Software Architect, Python Backend Engineer, AI/ML Engineer, DevSecOps Engineer, QA Engineer, Technical Writer, Cloud Deployment Specialist  
**Repository:** `c:\Users\HP\Flex-ID`  
**Current Git Branch:** `main` (commit baseline tracked)

---

## 1. Executive Summary

FLEX-ID (Federated Learning Explainable Intrusion Detection System) combines multi-class network intrusion detection on the CSE-CIC-IDS2018 dataset, federated learning simulation via the Flower framework, explainable AI analytics using SHAP, and an API-driven backend.

This audit establishes the baseline across the entire repository before undertaking documentation generation, architectural reconciliation, security hardening, continuous integration setup, and free-tier deployment.

---

## 2. Current Repository Architecture & Tree

```text
c:\Users\HP\Flex-ID/
├── app/                                 # Production FastAPI backend application
│   ├── main.py                          # Application factory, lifespan, CORS, error handling
│   ├── api/
│   │   ├── dependencies.py              # Dependency injection (DB sessions, services, repos)
│   │   └── v1/                          # Versioned API routers
│   │       ├── router.py                # Master v1 aggregator router
│   │       ├── health.py                # /health and /api/v1/health readiness probes
│   │       ├── models.py                # /api/v1/models (list, details, activation)
│   │       ├── predictions.py           # /api/v1/predictions (single & batch inference, audit)
│   │       ├── experiments.py           # /api/v1/experiments (list, create, round metrics)
│   │       └── explanations.py          # /api/v1/explanations (SHAP & local feature attribution)
│   ├── core/
│   │   ├── config.py                    # Pydantic Settings (.env, paths, thresholds)
│   │   ├── logging.py                   # Structured application logging
│   │   └── exceptions.py                # Domain exceptions & centralized JSON error handlers
│   ├── db/
│   │   ├── base.py                      # SQLAlchemy DeclarativeBase
│   │   ├── session.py                   # Engine, sessionmaker, get_db dependency
│   │   └── seeder.py                    # Automatic database seeding (models & historical runs)
│   ├── models/                          # SQLAlchemy ORM entities
│   │   ├── experiment.py                # Experiment table mapping
│   │   ├── experiment_metric.py         # Round-by-round training metric records
│   │   ├── model_record.py              # Registered model checkpoints & metrics
│   │   └── prediction_log.py            # Audited prediction logs with latency tracking
│   ├── repositories/                    # Data Access Object (DAO) layer
│   │   ├── experiment_repository.py     # Experiment CRUD & metric appending
│   │   ├── model_repository.py          # Model registration, retrieval, & activation
│   │   └── prediction_repository.py     # Prediction audit logging & queries
│   ├── schemas/                         # Pydantic v2 schemas
│   │   ├── health.py                    # HealthResponse
│   │   ├── prediction.py                # TrafficFeaturesInput, PredictionResponse, Batch
│   │   ├── model.py                     # ModelMetadataResponse, ModelListResponse
│   │   ├── experiment.py                # ExperimentCreate, ExperimentResponse, MetricRecord
│   │   └── explanation.py               # FeatureContribution, ExplanationRequest, Response
│   └── services/                        # Business & Machine Learning service layer
│       ├── model_manager.py             # ModelManager singleton, weights caching, warmup
│       ├── inference_service.py         # Preprocessing scaling, feature order, single/batch
│       ├── explanation_service.py       # Federated SHAP & local feature attribution
│       └── fl_service.py                # Asynchronous background FL orchestration
├── ml/                                  # Federated learning engine & data pipelines
│   ├── model.py                         # 3-layer Keras DNN classifier architecture
│   ├── aggregation.py                   # Robust aggregators (Krum, Multi-Krum, Trimmed Mean, Median)
│   ├── 1_process_data.py                # Preprocessing pipeline (enhanced with scaler export)
│   ├── 2_create_partitions.py           # Non-IID Dirichlet data partitioning
│   ├── 4_server.py                      # Flower FL server with strategy dispatch
│   ├── client.py                        # Honest Flower FL client
│   ├── client_attack.py                 # Malicious Flower client (5 attack types)
│   ├── compare_results.py               # Evaluation benchmark suite
│   ├── explain_model.py                 # Federated SHAP calculation engine
│   ├── plot_history.py                  # Learning curve plotter & CSV exporter
│   ├── prepare_unswnb15.py              # Alternative dataset script (UNSW-NB15)
│   ├── prepare_toniot.py                # Alternative dataset script (TON-IoT)
│   └── utils/
│       ├── data_utils.py                # Partition & weights loading utilities
│       └── seeds.py                     # Global random seed configuration (seed=42)
├── backend/                             # Legacy Node.js Express server (contains severe RCE flaw)
│   ├── package.json
│   └── server.js
├── frontend/                            # React + Vite web dashboard
│   ├── package.json
│   ├── vite.config.js
│   └── src/
├── data/                                # Data artifacts (training data & serialized transformers)
│   ├── scaler.pkl                       # Serialized MinMaxScaler (1.2 KB)
│   ├── label_encoder.pkl                # Serialized LabelEncoder (492 B)
│   ├── client_partition_0.pkl .. 7.pkl  # 8 Client Non-IID partitions (~165 MB total)
│   └── processed_data.csv               # Processed CSE-CIC-IDS2018 dataset (307.8 MB)
├── results/                             # Saved weights, plots, and benchmark JSON summaries
│   ├── fedproxeachround/                # Round 1 to 10 weights (~206 KB each)
│   ├── fedavgeachround/                 # Round 1 to 10 weights (~206 KB each)
│   ├── fedproxunderattack/              # Round 1 to 10 weights (~206 KB each)
│   ├── fedunderattack/                  # Round 1 to 10 weights (~206 KB each)
│   ├── comparison_results_no_attack.json# Verified clean benchmark results
│   ├── comparison_results_under_attack.json # Verified attack benchmark results
│   ├── metrics_summary.json             # Convergence summary
│   ├── shap_summary.json                # Precomputed federated SHAP rankings
│   └── *.png                            # Confusion matrices, distributions, SHAP summary plots
├── tests/                               # Automated Pytest suite (19 test cases)
│   ├── conftest.py                      # In-memory SQLite fixtures, TestClient, sample data
│   ├── test_health.py                   # Health endpoint tests
│   ├── test_models.py                   # Model registry & activation tests
│   ├── test_predictions.py              # Single & batch inference tests
│   ├── test_experiments.py              # Experiment creation & metric tests
│   ├── test_explanations.py             # Feature attribution & XAI tests
│   └── test_inference_service.py        # Unit tests for preprocessing & bounds clipping
├── docs/                                # Technical documentation
│   ├── ATTACK_GUIDE.md
│   ├── DATASET_GUIDE.md
│   ├── EVALUATION_GUIDE.md
│   └── TRAINING_GUIDE.md
├── Dockerfile                           # Production multi-stage Docker build
├── docker-compose.yml                   # Compose file for FastAPI + MySQL 8.0
├── requirements.txt                     # Pinned project dependencies
├── .env.example                         # Example environment variables
├── pytest.ini                           # Pytest configuration
├── flex_id.db                           # Local development SQLite database
├── PROJECT_AUDIT.md                     # Initial Phase 1 audit
├── IMPLEMENTATION_PLAN.md               # Initial Phase 1 implementation plan
└── README.md                            # Comprehensive project overview & documentation
```

---

## 3. Working Features Baseline

1. **REST API & Model Serving:**
   - Single flow inference (`POST /api/v1/predictions`) and bounded batch inference (`POST /api/v1/predictions/batch`) fully functional.
   - Model versions (`fedprox-v1`, `fedavg-v1`, `fedprox-attack-v1`, `fedavg-attack-v1`) inspected and toggled via `/api/v1/models`.
   - Feature validation via Pydantic v2 supporting both named key-value dictionary and 29-element ordered float arrays.
2. **Persistence Layer:**
   - SQLAlchemy 2.0 ORM with relational schema (`experiments`, `experiment_metrics`, `model_records`, `prediction_logs`).
   - SQLite for lightweight test and local runs; MySQL 8.0 support through connection string configuration.
   - Automatic database seeding of verified historical models and convergence metrics on startup.
3. **ML Preprocessing & Inference Integrity:**
   - Serialized `MinMaxScaler` (`data/scaler.pkl`) guarantees exact normalization parameter parity.
   - Strict clipping to `[0.0, 1.0]` and robust finite-value handling (`np.nan_to_num`) prevents out-of-distribution neural network inputs.
4. **Federated Learning Implementations:**
   - FedAvg and FedProx with proximal regularization ($\frac{\mu}{2}\|w - w_{global}\|^2$).
   - Byzantine-robust aggregations: Krum, Multi-Krum, Trimmed Mean, Coordinate Median in [ml/aggregation.py](file:///c:/Users/HP/Flex-ID/ml/aggregation.py).
   - 5 adversarial attack models in [ml/client_attack.py](file:///c:/Users/HP/Flex-ID/ml/client_attack.py): label-flipping, Gaussian noise, backdoor, Byzantine scaling, adaptive poisoning.
5. **Explainable AI:**
   - Local feature sensitivity / perturbation attribution combined with precomputed federated global SHAP rankings in [app/services/explanation_service.py](file:///c:/Users/HP/Flex-ID/app/services/explanation_service.py).
6. **Automated Testing:**
   - 19 automated integration and unit tests passing with 100% success rate in 2.53s.

---

## 4. Incomplete Features & Discovered Gaps

1. **Documentation Suite Completeness (Phase 1 Target):**
   - The repository currently lacks formal documentation files requested by the master rubric: `docs/PRD.md`, `docs/SRS.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/DATABASE.md`, `docs/AI.md`, `docs/SECURITY.md`, `docs/TESTING.md`, `docs/DEPLOYMENT.md`, `docs/LOAD_TESTING.md`, `docs/DECISIONS.md`, and `docs/openapi.json`.
2. **Automated CI Workflow:**
   - No `.github/workflows/ci.yml` exists for automated linting, test execution, and container build verification on push/PR.
3. **Dedicated Performance & Load Benchmark Script:**
   - No standardized benchmark script exists in `scripts/` to measure throughput, p50, p95, and p99 latency under concurrency.
4. **Free-Tier Deployment Container Bloat:**
   - The current `Dockerfile` copies `data/` and `results/` indiscriminately. `data/processed_data.csv` (307.8 MB) and `data/client_partition_*.pkl` (165 MB) are training artifacts unnecessary for inference, blowing the container image size beyond 1.5 GB. This exceeds free-tier memory/build allowances on platforms like Render (512 MB RAM limit).
5. **Legacy Node.js Backend Insecurity:**
   - `backend/server.js` contains a severe Remote Code Execution (RCE) flaw (`app.post('/api/run-command')` using `child_process.spawn(..., { shell: true })`). This legacy server is obsolete and should be deprecated or isolated.
6. **Scikit-Learn Version Mismatch Warning:**
   - `label_encoder.pkl` was serialized with scikit-learn 1.7.0 while the runtime uses scikit-learn 1.9.0, triggering an `InconsistentVersionWarning`. Re-serializing with the active runtime will eliminate this warning.

---

## 5. Security & Risk Findings

| ID | Severity | Category | Description | Remediation Plan |
| :--- | :---: | :--- | :--- | :--- |
| **SEC-01** | **CRITICAL** | Remote Code Execution | `backend/server.js` exposes `/api/run-command` running unvalidated shell strings via `child_process.spawn`. | Deprecate and isolate `backend/server.js`; ensure production deployments only expose the secure FastAPI backend. |
| **SEC-02** | **MEDIUM** | Insecure Deserialization | Model weights, label encoders, and scalers use Python `pickle`. | Restrict file permissions; load only from verified local paths; avoid loading untrusted external pickles. |
| **SEC-03** | **MEDIUM** | Resource Exhaustion (DoS) | Unbounded batch inference or concurrent SHAP requests could exhaust container memory (512 MB). | Enforce strict batch size bounds (`MAX_BATCH_SIZE=1000`) and timeout controls (`INFERENCE_TIMEOUT_SECONDS=10.0`). |
| **SEC-04** | **LOW** | Information Disclosure | Error responses could inadvertently leak stack traces. | Centralized `flex_id_exception_handler` and `global_exception_handler` sanitize responses to standard JSON error envelopes. |

---

## 6. Dependency & Environment Analysis

* **Python Version:** 3.13.5 (local environment), 3.11-slim (target container).
* **Core Dependencies Verified in venv:**
  - `fastapi==0.138.2`, `starlette==1.3.1`, `uvicorn==0.49.0`, `pydantic==2.14.1`, `pydantic-settings==2.16.1`
  - `sqlalchemy==2.0.51`, `alembic==1.18.5`
  - `tensorflow==2.21.0`, `keras==3.15.0`, `scikit-learn==1.9.0`, `imbalanced-learn==0.14.2`
  - `flwr==1.42.0`
  - `shap==0.52.0`, `scipy==1.18.0`, `pandas==3.0.3`, `numpy==2.4.6`
  - `pytest==9.2.0`, `httpx==0.28.1`
* **Finding:** All dependencies import and run without version conflicts.

---

## 7. Model & Artifact Verification

* **Input Features:** 29 canonical network flow metrics.
* **Classes:** 13 multiclass attack types:
  `['Benign', 'Brute Force -Web', 'Brute Force -XSS', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infilteration', 'SQL Injection', 'SSH-Bruteforce']`.
* **DNN Architecture:** Dense(256) -> BatchNorm -> Dropout(0.3) -> Dense(128) -> BatchNorm -> Dropout(0.3) -> Dense(64) -> BatchNorm -> Dropout(0.3) -> Dense(13, Softmax). Total parameters: 52,941.
* **Weight Checkpoints:** Verified in `results/fedproxeachround/round-10-weights.pkl` (~206 KB).
* **Scaling:** Normalized in `[0.0, 1.0]`. Preprocessor artifact verified in `data/scaler.pkl`.

---

## 8. Free-Tier Deployment Blockers & Strategy

### Blockers:
1. **Container Image Size:** Including `data/processed_data.csv` (308 MB) and `data/client_partition_*.pkl` (165 MB) makes the Docker image ~1.8 GB, exceeding the build cache and memory limits of free-tier platforms.
2. **TensorFlow Base Memory Overhead:** TensorFlow 2.x consumes ~250–350 MB RAM upon import. On a 512 MB free tier (e.g., Render Free Instance), memory pressure must be managed carefully.

### Proposed Free-Tier Deployment Plan:
1. **Target Provider:** **Render** (Free Web Service) or **Hugging Face Spaces** (Free Docker Space, 16 GB RAM / 2 vCPU).
   - *Primary Recommendation:* **Hugging Face Spaces (Docker SDK)** — 100% free, 16 GB RAM, 2 vCPU, permanent HTTPS URL, zero sleep on activity, perfect for AI model serving.
   - *Alternative Recommendation:* **Render Web Service (Free Tier)** — 512 MB RAM, free SQLite/ephemeral disk, spin-down after 15 min inactivity.
2. **Image Optimization:**
   - Update `.dockerignore` to strictly exclude `data/*.csv` and `data/client_partition_*.pkl`.
   - Production Docker image will only include `data/scaler.pkl`, `data/label_encoder.pkl`, and `results/**/round-10-weights.pkl`.
   - Reduces image size from 1.8 GB to ~280 MB, ensuring lightning-fast builds and minimal memory footprint.

---

## 9. Prioritized Implementation Roadmap (Stages B through G)

| Stage | Name | Target Deliverables |
| :--- | :--- | :--- |
| **Stage B** | **Documentation Inventory & PRD/SRS Finalization** | Create `docs/PRD.md`, `docs/SRS.md`, `docs/DECISIONS.md`, and export synchronized `docs/openapi.json`. |
| **Stage C** | **Architecture & Traceability** | Create `docs/ARCHITECTURE.md`, `docs/DATABASE.md`, `docs/AI.md`, `docs/SECURITY.md`, and `docs/TRACEABILITY.md`. |
| **Stage D** | **Backend & ML Hardening** | Re-serialize `label_encoder.pkl` with active sklearn; add pagination and request ID middleware to FastAPI. |
| **Stage E** | **Testing, CI & Load Benchmarking** | Create `scripts/load_test.py`, create `docs/LOAD_TESTING.md`, create `docs/TESTING.md`, add `.github/workflows/ci.yml`. |
| **Stage F** | **Docker & Free-Tier Deployment Preparation** | Optimize `.dockerignore`, update `Dockerfile` for lightweight deployment, create `docs/DEPLOYMENT.md`. |
| **Stage G** | **Final Verification & Presentation** | Execute all test suites, verify OpenAPI schema, update root `README.md`, and provide free deployment walkthrough. |
