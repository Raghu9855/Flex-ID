# FLEX-ID: System Architecture & Technical Design

**Document Version:** 1.0.0  
**Status:** Approved  
**Scope:** Backend Architecture, ML Serving, Federated Orchestration, and Persistence  

---

## 1. System Context Diagram (C4 Level 1)

```mermaid
C4Context
    title System Context Diagram for FLEX-ID Platform

    Person(analyst, "Security Analyst / SOC", "Monitors network anomalies, investigates intrusion alerts, and inspects feature attribution.")
    Person(researcher, "ML / Security Researcher", "Runs federated learning benchmarks, evaluates adversarial defenses, and inspects convergence.")
    System_Ext(sensor, "Network Packet Sensor / Edge Gateway", "Extracts network flow statistics (29 features) and sends telemetry to API.")

    System(flexid, "FLEX-ID Platform", "FastAPI backend, ML model serving engine, federated learning coordinator, and explainability service.")

    SystemDb(db, "Relational Database", "SQLite (dev/test) / MySQL (production) storing experiments, models, and audited predictions.")
    System_Ext(artifacts, "Artifact Storage", "Serialized Keras weights (.pkl), MinMaxScaler (.pkl), and SHAP summaries.")

    Rel(sensor, flexid, "Submits network flow telemetry", "HTTPS / JSON")
    Rel(analyst, flexid, "Inspects models, reviews predictions, requests XAI", "HTTPS / JSON / Web UI")
    Rel(researcher, flexid, "Triggers FL experiments and monitors training", "HTTPS / JSON")
    Rel(flexid, db, "Persists experiments, metrics, models, and prediction logs", "SQLAlchemy ORM")
    Rel(flexid, artifacts, "Loads weights, scalers, and label encoders", "Local Filesystem")
```

---

## 2. Component & Container Diagram (C4 Level 2)

```mermaid
flowchart TD
    subgraph ClientLayer["Edge & Client Layer"]
        A[Network Sensor / Agent]
        B[Security Operations Analyst]
        C[Web Dashboard / Admin]
    end

    subgraph FastAPILayer["FastAPI Application (ASGI Worker)"]
        D["API Routers\n/health · /models · /predictions · /experiments · /explanations"]
        E["Pydantic v2 Validation & Exception Filter"]
        
        subgraph Services["Domain Services Layer"]
            F["InferenceService\n(Feature Normalization & Forward Pass)"]
            G["ModelManager (Singleton)\n(Keras Model Cache & Startup Warmup)"]
            H["ExplanationService\n(Federated SHAP & Sensitivity Perturbation)"]
            I["FederatedLearningService\n(Asynchronous Background FL Runner)"]
        end

        subgraph Repositories["Data Access Layer (Repository Pattern)"]
            J["ModelRepository"]
            K["PredictionRepository"]
            L["ExperimentRepository"]
        end
    end

    subgraph DataStorage["Storage & Artifacts Layer"]
        M[("Relational Database\nSQLite / MySQL 8.0")]
        N["data/scaler.pkl (MinMaxScaler)"]
        O["data/label_encoder.pkl (13 Classes)"]
        P["results/**/round-10-weights.pkl"]
        Q["results/shap_summary.json"]
    end

    ClientLayer -->|HTTP/REST| D
    D --> E
    E --> F & H & I
    F --> G
    G --> N & O & P
    H --> G & Q
    F & I --> J & K & L
    J & K & L --> M
```

---

## 3. End-to-End Data Flows

### 3.1 Real-Time Prediction Request Flow
```mermaid
sequenceDiagram
    autonumber
    actor Sensor as Sensor / Client
    participant Router as /api/v1/predictions
    participant Service as InferenceService
    participant MM as ModelManager
    participant Keras as DNN Model
    participant Repo as PredictionRepository
    participant DB as Relational DB

    Sensor->>Router: POST /api/v1/predictions (29 Features)
    Router->>Service: predict_single(TrafficFeaturesInput)
    Service->>Service: _extract_feature_vector() & sanitize NaNs
    Service->>Service: _preprocess_batch() via MinMaxScaler & clip [0, 1]
    Service->>MM: get_model("fedprox-v1")
    MM-->>Service: Cached Keras Model Reference
    Service->>Keras: model.predict(scaled_sample)
    Keras-->>Service: Class Softmax Probabilities
    Service->>Repo: create(prediction_id, predicted_class, confidence, latency)
    Repo->>DB: INSERT INTO prediction_logs
    Service-->>Router: PredictionResponse (class, is_intrusion, confidence, latency)
    Router-->>Sensor: HTTP 200 OK JSON
```

### 3.2 Asynchronous Federated Experiment Flow
```mermaid
sequenceDiagram
    autonumber
    actor Researcher as Researcher
    participant Router as /api/v1/experiments
    participant FLService as FederatedLearningService
    participant Thread as Background Daemon Thread
    participant Repo as ExperimentRepository
    participant DB as Relational DB

    Researcher->>Router: POST /api/v1/experiments (FedProx, 10 rounds, alpha=0.4)
    Router->>FLService: create_experiment(ExperimentCreate)
    FLService->>Repo: create(status="running")
    Repo->>DB: INSERT INTO experiments
    FLService->>Thread: Launch _run_simulation_background()
    FLService-->>Router: HTTP 202 Accepted (ExperimentResponse)
    Router-->>Researcher: Immediate response with experiment_id
    
    loop For Each Round (1..Rounds)
        Thread->>Thread: Compute client fit & aggregate weights
        Thread->>Repo: add_metric(round, train_loss, accuracy, f1)
        Repo->>DB: INSERT INTO experiment_metrics
    end
    Thread->>Repo: update_status("completed")
    Repo->>DB: UPDATE experiments SET status="completed"
```

---

## 4. Module Responsibilities & Separation of Concerns

1. **`app.api` (Transport Layer):** Validates HTTP contracts, maps query/path/body parameters, and returns status codes. Business logic is strictly excluded.
2. **`app.services` (Business Logic Layer):** Coordinates model execution, feature normalization, batching, perturbation calculations, and background simulation threads.
3. **`app.repositories` (Data Access Layer):** Encapsulates all raw SQLAlchemy queries, commits, and rollbacks. Decouples domain logic from SQL dialect differences.
4. **`app.models` & `app.schemas` (Contracts):**
   - `app.models`: SQLAlchemy ORM database table definitions.
   - `app.schemas`: Pydantic v2 schemas for request validation, serialization, and OpenAPI documentation.
5. **`ml` (Research & Training Pipeline):** Contains standalone Flower FL scripts, partition generators, and model architectures. Preserved intact for offline research and reproducible training.

---

## 5. Deployment Architecture & Trust Boundaries

```
[Untrusted Internet / Corporate LAN]
                │
                ▼ (Port 80 / 443 HTTPS)
   ┌─────────────────────────┐
   │ Reverse Proxy / Ingress │
   │   (Cloudflare / NGINX)  │
   └────────────┬────────────┘
                │ Port 8000 (HTTP)
                ▼ (TRUST BOUNDARY: Private VPC / Internal Container Network)
   ┌─────────────────────────────────────────────────────────────┐
   │                FLEX-ID Application Container                │
   │  Non-root user: appuser (UID 1001)                          │
   │  Uvicorn ASGI Server · FastAPI Application                  │
   └────────────────────────────┬────────────────────────────────┘
                                │ Port 3306 (TCP)
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  MySQL 8.0 Database Container               │
   │  Persistent Volume: mysql_data                             │
   └─────────────────────────────────────────────────────────────┘
```

### Trust Boundary Protections:
- **Container Isolation:** API runs as a non-privileged system user (`appuser`, UID 1001).
- **Network Boundaries:** Database port 3306 is not exposed to the public internet; accessible only over the internal bridge network.
- **Input Boundaries:** Strict Pydantic validation rejects un-canonical feature shapes before touching the neural network runtime.
