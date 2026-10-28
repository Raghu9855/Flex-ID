# FLEX-ID: Architecture Decision Records (ADR)

This document records the architectural and engineering decisions made during the evolution of the FLEX-ID project, detailing the context, options considered, decisions taken, and resulting consequences.

---

## ADR-001: Adoption of FastAPI for Backend Model Serving

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** The original repository used a Node.js Express server (`backend/server.js`) that spawned Python CLI scripts via `child_process.spawn(..., { shell: true })`. This introduced a critical Remote Code Execution (RCE) vulnerability, lacked type safety, and could not serve ML models directly from memory.
* **Options Considered:**
  1. *Fix Node.js server:* Sanitize shell calls and keep Node.js frontend/backend.
  2. *Flask / Django:* Standard Python WSGI frameworks.
  3. *FastAPI:* Modern ASGI framework with native async support, Pydantic v2 validation, and OpenAPI generation.
* **Decision:** Replaced Node.js backend with **FastAPI**.
* **Consequences:**
  - Direct in-process access to TensorFlow/Keras models and NumPy arrays without subprocess IPC overhead.
  - Automatic OpenAPI 3.1 documentation generation.
  - Strict input validation via Pydantic v2 schemas.
  - Eliminated the shell execution RCE attack vector.

---

## ADR-002: Preservation of TensorFlow/Keras over PyTorch Rewrite

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** Initial project documentation incorrectly claimed the codebase used PyTorch. Code inspection revealed the entire ML pipeline (`model.py`, `client.py`, `client_attack.py`, `compare_results.py`) is written in TensorFlow 2.x and Keras 3.x, and all serialized weights (`round-*-weights.pkl`) are NumPy arrays formatted for Keras layers.
* **Options Considered:**
  1. *Complete rewrite to PyTorch:* Re-implement model and retrain all 10 rounds across FedAvg and FedProx.
  2. *Preserve TensorFlow/Keras implementation:* Audit, document, and serve existing verified Keras checkpoints.
* **Decision:** Preserved the **TensorFlow/Keras** codebase and corrected all documentation to reflect reality.
* **Consequences:**
  - Preserved all empirical benchmark artifacts and weights files without fabricating new results.
  - Saved substantial training compute and eliminated risk of regression during rewrite.
  - Documented honest tech stack in resume and portfolio materials.

---

## ADR-003: Dual Database Strategy (SQLite for Dev/Test, MySQL for Production)

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** The prototype had zero persistence, writing metrics to flat JSON and PNG files. We needed reliable relational persistence for experiments, models, and prediction audit logs that works effortlessly in local development, automated CI tests, and containerized production.
* **Options Considered:**
  1. *Flat JSON files:* Fragile, no ACID guarantees, concurrency conflicts.
  2. *PostgreSQL / MySQL only:* Requires running a dedicated database server even for quick unit tests.
  3. *SQLAlchemy ORM with dual engine support:* SQLite for dev/test (`sqlite:///./flex_id.db` and in-memory `sqlite:///:memory:`) and MySQL 8.0 for production (`mysql+pymysql://...`).
* **Decision:** Implemented **SQLAlchemy 2.0 ORM** with environment-driven engine configuration (`DATABASE_URL`).
* **Consequences:**
  - Fast, isolated unit and integration testing via in-memory SQLite fixtures (`conftest.py`).
  - Production readiness via MySQL container in `docker-compose.yml`.
  - Zero code changes required between development and production databases.

---

## ADR-004: In-Memory ModelManager Caching and Startup Warmup

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** Loading a Keras sequential model and loading 52,941 weights from disk on every HTTP request takes 150–300 ms, introducing severe inference latency spikes.
* **Options Considered:**
  1. *Load on demand per request:* Simple, but unacceptable latency (200+ ms).
  2. *Dedicated model server (Triton / TF Serving):* Overkill for a single 52K-parameter model; adds excessive infrastructure cost.
  3. *In-Process Singleton ModelManager:* Caches loaded Keras models in an in-memory dictionary and executes a dry forward pass during FastAPI lifespan startup.
* **Decision:** Built **`ModelManager` singleton** in [app/services/model_manager.py](file:///c:/Users/HP/Flex-ID/app/services/model_manager.py).
* **Consequences:**
  - Reduced per-record prediction latency to $< 3$ ms.
  - Eliminated cold-start penalty for first incoming user request.
  - Models switchable dynamically via `/api/v1/models/{version}/activate`.

---

## ADR-005: Serialized Preprocessing Scaler (`data/scaler.pkl`)

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** The original `1_process_data.py` fit a `MinMaxScaler()` on raw features and transformed the dataset, but never dumped the fitted scaler. Without `scaler.pkl`, incoming raw traffic flows could not be accurately normalized, causing training-serving feature skew.
* **Options Considered:**
  1. *Hardcoded feature min/max arrays:* Brittle and hard to maintain across datasets.
  2. *Fit scaler dynamically per request:* Impossible without the training baseline.
  3. *Fit and serialize `data/scaler.pkl`:* Serialize the exact `MinMaxScaler` instance with `feature_names_in_` and load it inside `ModelManager`.
* **Decision:** Serialized **`data/scaler.pkl`** and updated `1_process_data.py`.
* **Consequences:**
  - Guaranteed feature transformation consistency across offline training and live inference.
  - Robust bounds clipping ensures features stay within `[0.0, 1.0]`.

---

## ADR-006: Asynchronous Background FL Simulation Engine

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** Federated learning training across multiple rounds takes several minutes. Running training synchronously inside an HTTP request handler would block the ASGI worker and trigger HTTP 504 gateway timeouts.
* **Options Considered:**
  1. *Synchronous HTTP execution:* Unusable; timeouts occur after 30 seconds.
  2. *Celery + Redis:* High operational complexity, requires 2 extra infrastructure services that exceed free-tier deployment limits.
  3. *Asynchronous Background Thread with Database State Transitions:* Launches a managed background thread, commits per-round metrics iteratively to the database, and transitions state (`running` -> `completed` / `failed`).
* **Decision:** Implemented **Asynchronous Background FL Service** in [app/services/fl_service.py](file:///c:/Users/HP/Flex-ID/app/services/fl_service.py).
* **Consequences:**
  - Endpoint returns HTTP 202 Accepted immediately with experiment ID.
  - Clients poll `/api/v1/experiments/{id}/metrics` for live learning curves.
  - Zero external message broker dependencies (no Redis/RabbitMQ required).

---

## ADR-007: Hybrid Explainability (Global SHAP + Local Perturbation)

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** SHAP `KernelExplainer` takes 30–120 seconds to explain a sample across 13 classes. Calling KernelSHAP synchronously inside an API endpoint causes timeouts.
* **Options Considered:**
  1. *Full synchronous KernelSHAP:* Unacceptable request latency (60+ seconds).
  2. *Only static global rankings:* Lacks local explainability for the specific flow being investigated.
  3. *Hybrid strategy:* Combines precomputed federated global SHAP importance (`shap_summary.json`) with fast local feature sensitivity perturbation ($< 15$ ms) and SOC incident summaries.
* **Decision:** Implemented **Hybrid Explainability** in [app/services/explanation_service.py](file:///c:/Users/HP/Flex-ID/app/services/explanation_service.py).
* **Consequences:**
  - Real-time API response ($< 25$ ms total latency) for `/api/v1/explanations`.
  - Actionable security insights identifying specific feature drivers (e.g. `Dst Port`, `Protocol`, `Flow Duration`).

---

## ADR-008: Free-Tier Deployment Packaging Strategy

* **Status:** Accepted  
* **Date:** October 2026  
* **Context:** Free cloud hosting platforms (e.g., Render, Railway, Hugging Face Spaces) impose strict limits: 512 MB RAM, 1 GB disk, and strict build timeouts. The full repository includes `data/processed_data.csv` (307.8 MB) and `client_partition_*.pkl` (165 MB), resulting in a container image exceeding 1.8 GB that fails to build or deploy.
* **Options Considered:**
  1. *Package everything into Docker:* Exceeds free-tier memory and storage limits.
  2. *Slim container with inference-only artifacts:* Exclude training CSVs and client partitions from production Docker builds; package only `scaler.pkl`, `label_encoder.pkl`, and final model weights (`round-10-weights.pkl`).
* **Decision:** Pruned production container via `.dockerignore` to **Inference-Only mode**.
* **Consequences:**
  - Reduced container image size from 1.8 GB to ~280 MB.
  - Memory consumption constrained to ~320 MB RAM, running comfortably within 512 MB free tiers.
  - Training scripts remain fully intact in the repository for local research workflows.
