# FLEX-ID: API Reference & Endpoint Specification

**Document Version:** 1.0.0  
**Base URL:** `http://localhost:8000` (or host domain)  
**API Prefix:** `/api/v1`  
**OpenAPI Specification:** [docs/openapi.json](file:///c:/Users/HP/Flex-ID/docs/openapi.json)  
**Interactive UI:** `http://localhost:8000/docs` (Swagger UI)  

---

## 1. Authentication & Common Headers

* **Authentication:** Currently unauthenticated for MVP / internal microservice mesh.
* **Content-Type:** `application/json` (Required for all `POST` payloads).
* **Accept:** `application/json`.
* **Standard Error Envelope:**
  ```json
  {
    "error": "ErrorClassName",
    "message": "Human-readable explanation of error.",
    "path": "/api/v1/resource"
  }
  ```

---

## 2. Health & System Status

### `GET /health` / `GET /api/v1/health`
* **Summary:** Application Readiness & Health Probe.
* **Response Codes:**
  - `200 OK`: All components operational.
  - `503 Service Unavailable`: Critical subsystem failure (e.g. database unreachable).
* **Response Body:**
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-10-28T05:00:00.000Z",
    "environment": "development",
    "version": "1.0.0",
    "database_connected": true,
    "active_model": "fedprox-v1",
    "components": {
      "database": "reachable",
      "model_manager": "ready",
      "classes_loaded": 13,
      "scaler_loaded": true,
      "models_cached": ["fedprox-v1"]
    }
  }
  ```

---

## 3. Models & Serving Management

### `GET /api/v1/models`
* **Summary:** List registered model versions and baseline evaluation metrics.
* **Response Body:**
  ```json
  {
    "total_models": 4,
    "active_model": "fedprox-v1",
    "models": [
      {
        "id": "e9a03b57-6101-4475-b9f1-a12b48902cd3",
        "model_version": "fedprox-v1",
        "strategy": "fedprox",
        "weights_path": "results/fedproxeachround/round-10-weights.pkl",
        "input_features_count": 29,
        "num_classes": 13,
        "accuracy": 0.9093,
        "macro_f1": 0.5899,
        "weighted_f1": 0.9109,
        "is_active": true,
        "created_at": "2026-10-28T04:45:00Z",
        "class_names": ["Benign", "Brute Force -Web", "DDOS attack-HOIC", "..."]
      }
    ]
  }
  ```

### `GET /api/v1/models/{model_id_or_version}`
* **Summary:** Get metadata and metrics for a specific model version.
* **Path Parameters:**
  - `model_id_or_version` (string, required): e.g. `fedprox-v1` or UUID.
* **Status Codes:** `200 OK`, `404 Not Found`.

### `POST /api/v1/models/{model_version}/activate`
* **Summary:** Dynamically sets target model version as default for live traffic inference.
* **Status Codes:** `200 OK`, `404 Not Found`.

---

## 4. Intrusion Detection & Inference

### `POST /api/v1/predictions`
* **Summary:** Evaluate single network flow record for intrusion patterns.
* **Request Body (Dictionary Format):**
  ```json
  {
    "features": {
      "Dst Port": 80.0,
      "Protocol": 6.0,
      "Flow Duration": 125000.0,
      "Tot Fwd Pkts": 10.0,
      "Tot Bwd Pkts": 8.0,
      "TotLen Fwd Pkts": 500.0,
      "TotLen Bwd Pkts": 1200.0,
      "Fwd Pkt Len Max": 200.0,
      "Fwd Pkt Len Mean": 50.0,
      "Bwd Pkt Len Max": 400.0,
      "Bwd Pkt Len Mean": 150.0,
      "Flow Byts/s": 13600.0,
      "Flow Pkts/s": 144.0,
      "Flow IAT Mean": 8500.0,
      "Flow IAT Max": 25000.0,
      "Fwd IAT Mean": 12000.0,
      "Bwd IAT Mean": 15000.0,
      "Fwd Header Len": 200.0,
      "Bwd Header Len": 160.0,
      "Fwd Pkts/s": 80.0,
      "Bwd Pkts/s": 64.0,
      "Pkt Len Mean": 94.4,
      "Pkt Len Max": 400.0,
      "Pkt Len Var": 15200.0,
      "SYN Flag Cnt": 0.0,
      "RST Flag Cnt": 0.0,
      "ACK Flag Cnt": 1.0,
      "Init Fwd Win Byts": 26883.0,
      "Init Bwd Win Byts": 26883.0
    },
    "model_version": "fedprox-v1"
  }
  ```
* **Alternative Request Body (Array Format):**
  ```json
  {
    "features": [80.0, 6.0, 125000.0, 10.0, 8.0, 500.0, 1200.0, 200.0, 50.0, 400.0, 150.0, 13600.0, 144.0, 8500.0, 25000.0, 12000.0, 15000.0, 200.0, 160.0, 80.0, 64.0, 94.4, 400.0, 15200.0, 0.0, 0.0, 1.0, 26883.0, 26883.0]
  }
  ```
* **Response Body (`200 OK`):**
  ```json
  {
    "prediction_id": "c1f7a4e0-3942-491b-832f-a9cb89d91f24",
    "model_version": "fedprox-v1",
    "predicted_class": "Benign",
    "predicted_class_id": 0,
    "is_intrusion": false,
    "confidence": 0.9842,
    "class_probabilities": {
      "Benign": 0.9842,
      "DoS attacks-Hulk": 0.0091,
      "DDOS attack-HOIC": 0.0043,
      "FTP-BruteForce": 0.0011
    },
    "latency_ms": 2.14
  }
  ```

### `POST /api/v1/predictions/batch`
* **Summary:** Concurrently classifies an array of up to 1,000 flow records.
* **Request Body:**
  ```json
  {
    "records": [
      { "features": [ ... 29 floats ... ] },
      { "features": { ... 29 key-value pairs ... } }
    ],
    "model_version": "fedprox-v1"
  }
  ```
* **Response Body (`200 OK`):**
  ```json
  {
    "model_version": "fedprox-v1",
    "total_processed": 2,
    "intrusions_detected": 0,
    "predictions": [ ... list of PredictionResponse objects ... ],
    "total_latency_ms": 4.12
  }
  ```
* **Status Codes:**
  - `200 OK`: Successfully processed.
  - `400 Bad Request`: Payload exceeds 1,000 records (`BatchSizeExceededError`).
  - `422 Unprocessable Entity`: Schema validation error.

### `GET /api/v1/predictions/{prediction_id}`
* **Summary:** Retrieve audited prediction log from database.
* **Status Codes:** `200 OK`, `404 Not Found`.

---

## 5. Federated Learning Orchestration

### `POST /api/v1/experiments`
* **Summary:** Create and launch background federated learning simulation.
* **Request Body:**
  ```json
  {
    "name": "FedProx Resilience Test",
    "strategy": "fedprox",
    "rounds": 10,
    "num_clients": 4,
    "proximal_mu": 0.1,
    "is_attack": false
  }
  ```
* **Response Body (`202 Accepted`):**
  ```json
  {
    "id": "7058c381-cfe0-4b7d-95f0-8b11a8582a60",
    "name": "FedProx Resilience Test",
    "strategy": "fedprox",
    "rounds": 10,
    "num_clients": 4,
    "proximal_mu": 0.1,
    "is_attack": false,
    "status": "running",
    "created_at": "2026-10-28T05:00:00Z"
  }
  ```

### `GET /api/v1/experiments`
* **Summary:** List past and ongoing experiments.

### `GET /api/v1/experiments/{id}/metrics`
* **Summary:** Retrieve per-round learning curves.
* **Response Body (`200 OK`):**
  ```json
  [
    { "round_number": 1, "train_loss": 0.8123, "accuracy": 0.7245, "f1_score": 0.4210 },
    { "round_number": 2, "train_loss": 0.6512, "accuracy": 0.7891, "f1_score": 0.4850 },
    { "round_number": 10, "train_loss": 0.3389, "accuracy": 0.9093, "f1_score": 0.5899 }
  ]
  ```

---

## 6. Explainable AI (XAI) Analytics

### `POST /api/v1/explanations`
* **Summary:** Generate local sensitivity feature attribution and global SHAP summary for a flow vector.
* **Request Body:**
  ```json
  {
    "features": [ ... 29 floats ... ],
    "model_version": "fedprox-v1",
    "top_k": 3
  }
  ```
* **Response Body (`200 OK`):**
  ```json
  {
    "model_version": "fedprox-v1",
    "predicted_class": "Benign",
    "is_intrusion": false,
    "confidence": 0.9785,
    "method": "Federated-SHAP & Local Attribution",
    "top_contributing_features": [
      { "feature": "Protocol", "importance": 0.142, "direction": "positive" },
      { "feature": "Dst Port", "importance": 0.0984, "direction": "positive" },
      { "feature": "ACK Flag Cnt", "importance": 0.0512, "direction": "positive" }
    ],
    "global_importance_baseline": {
      "Protocol": 100.0,
      "Dst Port": 83.32,
      "ACK Flag Cnt": 51.5
    },
    "summary": "Traffic classified as Benign (normal operation, confidence 97.9%). Flow metrics conform to standard non-adversarial baselines."
  }
  ```
