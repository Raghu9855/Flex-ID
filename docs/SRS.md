# FLEX-ID: Software Requirements Specification (SRS)

**Document Version:** 1.0.0  
**Status:** Approved Baseline  
**Reference Document:** [docs/PRD.md](file:///c:/Users/HP/Flex-ID/docs/PRD.md)  

---

## 1. System Overview

FLEX-ID provides a modular, RESTful backend service for multi-class network intrusion detection, model lifecycle management, federated experiment tracking, and explainable AI analytics. The system exposes standard HTTP/JSON endpoints adhering to OpenAPI 3.1 specifications.

---

## 2. Actors & System Roles

1. **Client / Sensor Agent:** Automated network flow collection system that submits flow vectors to `/api/v1/predictions`.
2. **Security Analyst (Human / Dashboard):** User investigating alerts, reviewing model versions, and querying feature attributions via `/api/v1/explanations`.
3. **MLOps / Administrator:** Operator triggering federated experiments and switching active serving models.
4. **FastAPI Application (System):** Orchestrator executing input validation, scaling, inference, and persistence.
5. **Database Engine (System):** Persistent relational store (SQLite/MySQL) tracking state and audit trails.

---

## 3. Detailed Functional Requirements

### FR-01: Health & Readiness Probe
- **Path:** `GET /health` and `GET /api/v1/health`
- **Pre-conditions:** Application lifespan initialized.
- **Input:** None.
- **Processing:** Pings the active database via `SELECT 1`, queries `ModelManager` cache status, and checks scaler/label-encoder loading.
- **Output:** HTTP 200 with `HealthResponse` (`status`, `environment`, `database_connected`, `active_model`, `components`).
- **Post-conditions:** None.

### FR-02: Model Registry Listing
- **Path:** `GET /api/v1/models`
- **Input:** Optional pagination parameters (`skip`, `limit`).
- **Processing:** Queries `model_records` table via `ModelRepository`, retrieves active model flag, and attaches class names list.
- **Output:** HTTP 200 with `ModelListResponse` (`total_models`, `active_model`, `models`).

### FR-03: Model Details & Version Inspection
- **Path:** `GET /api/v1/models/{model_id_or_version}`
- **Input:** Path parameter `model_id_or_version` (e.g. `fedprox-v1` or UUID).
- **Processing:** Look up by `model_version`, fallback to `id`.
- **Output:** HTTP 200 with `ModelMetadataResponse`. If not found, HTTP 404 with standard error envelope.

### FR-04: Dynamic Model Activation
- **Path:** `POST /api/v1/models/{model_version}/activate`
- **Processing:** Atomically updates `model_records.is_active` to `False` for all models, sets target model to `True`, and pre-caches weights in `ModelManager`.
- **Output:** HTTP 200 with updated `ModelMetadataResponse`. If version invalid, HTTP 404.

### FR-05: Single Flow Prediction
- **Path:** `POST /api/v1/predictions`
- **Input:** `TrafficFeaturesInput` containing either:
  1. Dictionary with 29 required feature names (case-insensitive, whitespace/underscore agnostic).
  2. Array with exactly 29 numeric values in canonical order.
- **Validation Rules:**
  - Feature count must equal 29.
  - Infinite or NaN values coerced via `np.nan_to_num`.
  - Feature values scaled via `MinMaxScaler` and strictly clipped to `[0.0, 1.0]`.
- **Output:** HTTP 200 with `PredictionResponse` (`prediction_id`, `model_version`, `predicted_class`, `predicted_class_id`, `is_intrusion`, `confidence`, `class_probabilities`, `latency_ms`).
- **Post-conditions:** Persists audited record to `prediction_logs` table.

### FR-06: Bounded Batch Prediction
- **Path:** `POST /api/v1/predictions/batch`
- **Input:** `BatchTrafficInput` (`records: List[TrafficFeaturesInput]` where `1 <= len(records) <= 1000`).
- **Validation Rules:** If `len(records) > 1000`, reject with HTTP 400 (`BatchSizeExceededError`).
- **Processing:** Vectorizes feature matrix, executes single batched neural network forward pass, splits results.
- **Output:** HTTP 200 with `BatchPredictionResponse` (`total_processed`, `intrusions_detected`, `predictions`, `total_latency_ms`).

### FR-07: Prediction Audit Retrieval
- **Path:** `GET /api/v1/predictions/{prediction_id}`
- **Input:** Path parameter `prediction_id` (UUID).
- **Processing:** Queries `prediction_logs` table by primary key.
- **Output:** HTTP 200 with logged prediction details. If missing, HTTP 404.

### FR-08: Federated Experiment Creation
- **Path:** `POST /api/v1/experiments`
- **Input:** `ExperimentCreate` (`name`, `strategy`, `rounds`, `num_clients`, `proximal_mu`, `is_attack`, `attack_type`).
- **Validation Rules:** `rounds` between 1 and 100; `num_clients` between 2 and 32; `proximal_mu >= 0.0`.
- **Processing:** Creates record with status `"running"`, spawns background daemon thread to execute simulated FL rounds, commits round metrics iteratively.
- **Output:** HTTP 202 Accepted with `ExperimentResponse`.

### FR-09: Experiment Listing & Metric Inspection
- **Path:** `GET /api/v1/experiments` and `GET /api/v1/experiments/{id}/metrics`
- **Processing:** Returns paginated experiment records and per-round loss/accuracy/F1 progression.
- **Output:** HTTP 200 with `ExperimentListResponse` or `List[MetricRecord]`.

### FR-10: Explainable AI Feature Attribution
- **Path:** `POST /api/v1/explanations`
- **Input:** `ExplanationRequest` (`features`, `model_version`, `top_k: 1..29`).
- **Processing:** Executes base prediction, runs local feature sensitivity perturbation, maps to precomputed federated global SHAP ranking, and synthesizes SOC incident summary.
- **Output:** HTTP 200 with `ExplanationResponse` (`top_contributing_features`, `global_importance_baseline`, `summary`).

---

## 4. Input & Feature Canonical Specification

The 29 canonical features extracted from the CSE-CIC-IDS2018 dataset and validated across the pipeline:

| Index | Feature Name | Description | Typical Range |
| :---: | :--- | :--- | :--- |
| 0 | `Dst Port` | Destination port of flow | 0 – 65535 |
| 1 | `Protocol` | IP protocol number (TCP=6, UDP=17) | 0 – 255 |
| 2 | `Flow Duration` | Duration of flow in microseconds | 0 – 120,000,000 |
| 3 | `Tot Fwd Pkts` | Total packets in forward direction | 0 – 300,000 |
| 4 | `Tot Bwd Pkts` | Total packets in backward direction | 0 – 300,000 |
| 5 | `TotLen Fwd Pkts` | Total bytes in forward direction | 0 – 100,000,000 |
| 6 | `TotLen Bwd Pkts` | Total bytes in backward direction | 0 – 100,000,000 |
| 7 | `Fwd Pkt Len Max` | Maximum packet length forward | 0 – 65535 |
| 8 | `Fwd Pkt Len Mean`| Mean packet length forward | 0 – 65535 |
| 9 | `Bwd Pkt Len Max` | Maximum packet length backward | 0 – 65535 |
| 10 | `Bwd Pkt Len Mean`| Mean packet length backward | 0 – 65535 |
| 11 | `Flow Byts/s` | Flow throughput in bytes per second | 0 – 100,000,000 |
| 12 | `Flow Pkts/s` | Flow throughput in packets per second| 0 – 1,000,000 |
| 13 | `Flow IAT Mean` | Mean time between two packets | 0 – 120,000,000 |
| 14 | `Flow IAT Max` | Maximum time between two packets | 0 – 120,000,000 |
| 15 | `Fwd IAT Mean` | Mean time between forward packets | 0 – 120,000,000 |
| 16 | `Bwd IAT Mean` | Mean time between backward packets| 0 – 120,000,000 |
| 17 | `Fwd Header Len` | Total header bytes forward | 0 – 10,000,000 |
| 18 | `Bwd Header Len` | Total header bytes backward | 0 – 10,000,000 |
| 19 | `Fwd Pkts/s` | Forward packets per second | 0 – 1,000,000 |
| 20 | `Bwd Pkts/s` | Backward packets per second | 0 – 1,000,000 |
| 21 | `Pkt Len Mean` | Mean packet length across flow | 0 – 65535 |
| 22 | `Pkt Len Max` | Maximum packet length across flow | 0 – 65535 |
| 23 | `Pkt Len Var` | Variance of packet lengths | 0 – 1,000,000,000 |
| 24 | `SYN Flag Cnt` | Count of packets with SYN flag | 0 – 1 |
| 25 | `RST Flag Cnt` | Count of packets with RST flag | 0 – 1 |
| 26 | `ACK Flag Cnt` | Count of packets with ACK flag | 0 – 1 |
| 27 | `Init Fwd Win Byts`| Window bytes in forward direction | 0 – 65535 |
| 28 | `Init Bwd Win Byts`| Window bytes in backward direction | 0 – 65535 |

---

## 5. Non-Functional Requirements & Constraints

* **Performance:** Per-sample inference latency $< 10$ ms. Warmup routine executed on startup.
* **Memory Footprint:** Application must run within $< 450$ MB RAM to remain safely below the 512 MB free-tier limit.
* **Error Envelope Specification:**
  ```json
  {
    "error": "ModelNotFoundError",
    "message": "Model 'fedprox-v99' was not found.",
    "path": "/api/v1/models/fedprox-v99"
  }
  ```
* **Statelessness:** API container is stateless; persistent data resides in SQLite file (or external MySQL instance).
