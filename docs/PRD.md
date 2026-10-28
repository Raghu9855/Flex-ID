# FLEX-ID: Product Requirements Document (PRD)

**Document Version:** 1.0.0  
**Product Name:** FLEX-ID (Federated Learning Explainable Intrusion Detection System)  
**Status:** Approved / Active Baseline  
**Target Roles:** Backend Engineers, AI/ML Engineers, Security Analysts, DevOps Engineers  

---

## 1. Executive Summary

FLEX-ID is an API-driven, privacy-preserving cybersecurity platform that detects multi-class network traffic intrusions across distributed enterprise network nodes. Powered by Federated Learning (FL) via the Flower framework and served through a production-grade FastAPI REST service with SQLAlchemy persistence and SHAP explainability, FLEX-ID bridges academic federated learning research and practical, reliable backend engineering.

---

## 2. Problem Statement

1. **Regulatory & Privacy Barriers (GDPR / HIPAA):** Centralizing raw network packet streams and flow telemetry from multiple corporate entities or internal subnetworks violates privacy and confidentiality compliance.
2. **Data Heterogeneity (Non-IID Drift):** Traffic distributions vary significantly across different organizations (e.g., higher attack frequency in DMZ nodes vs. internal LANs), causing conventional centralized models or naïve federated averaging (FedAvg) to experience catastrophic gradient divergence.
3. **Black-Box AI Impasse:** Opaque neural network predictions cannot be verified or triaged by Security Operations Center (SOC) analysts without interpretable feature attribution.
4. **Academic Prototype Fragility:** Many FL projects exist only as disjointed CLI scripts, lacking model serving APIs, request validation, database audit trails, containerization, and automated test coverage.

---

## 3. Product Vision & Value Proposition

**Vision:** To provide a production-ready, lightweight, API-driven intrusion detection platform that trains collaborative AI models without raw data movement, serves low-latency inferences with full input validation, and translates neural network activations into human-readable security incident explanations.

---

## 4. Target Users & Use Cases

| User Persona | Role | Primary Use Cases |
| :--- | :--- | :--- |
| **SOC Security Analyst** | Triage & Threat Response | Submits suspicious network flow telemetry to `/api/v1/predictions`; reviews top contributing features via `/api/v1/explanations` to make firewall block decisions. |
| **SecOps / Platform Engineer** | Infrastructure & Monitoring | Inspects health readiness probes via `/health`; tracks latency, active model versions, and persistent audit logs in SQLite/MySQL. |
| **ML / Security Researcher** | Experimentation & Benchmarking| Configures and triggers multi-round federated training experiments via `/api/v1/experiments`; monitors convergence curves and tests adversarial attack resilience. |

---

## 5. Product Goals & Non-Goals

### Product Goals
- **G-01:** Deliver sub-millisecond per-record inference for 13 multiclass intrusion categories on CSE-CIC-IDS2018.
- **G-02:** Provide training-serving feature parity through serialized MinMax normalization (`data/scaler.pkl`).
- **G-03:** Support non-blocking federated learning simulation and experiment tracking with round-level metric persistence.
- **G-04:** Provide local feature attribution and global federated SHAP rankings on demand.
- **G-05:** Maintain 100% test pass rate across automated unit and integration tests.
- **G-06:** Support zero-cost / free-tier container deployment (memory footprint < 512 MB).

### Non-Goals
- **NG-01:** Physical cross-cloud node mesh orchestration (FLEX-ID provides algorithmically sound logical client simulations on a single workstation/server; it does not replace Kubernetes-based distributed cross-silo networking).
- **NG-02:** Live raw packet sniffing (libpcap/eBPF capture is upstream; FLEX-ID operates on extracted 29-feature flow records).
- **NG-03:** Complex enterprise RBAC/SSO (authentication is designed for API-key / internal microservice mesh integration; complex LDAP/ActiveDirectory integration is out of scope for MVP).

---

## 6. Functional Requirements (FR)

| Identifier | Requirement Name | Description | Status |
| :--- | :--- | :--- | :---: |
| **FR-01** | **Health & Readiness Probe** | `GET /health` and `GET /api/v1/health` must report system status, database connectivity, and model manager readiness. | **Implemented** |
| **FR-02** | **Model Registry Inspection** | `GET /api/v1/models` must list registered model versions, accuracy, F1-scores, and the currently active model. | **Implemented** |
| **FR-03** | **Model Detail Retrieval** | `GET /api/v1/models/{version}` must return model metadata, parameter dimensions, and file paths. | **Implemented** |
| **FR-04** | **Model Dynamic Activation** | `POST /api/v1/models/{version}/activate` must switch the active inference model without restarting the server. | **Implemented** |
| **FR-05** | **Single Flow Prediction** | `POST /api/v1/predictions` must validate 29 flow features, normalize them, and return predicted class, confidence, and latency. | **Implemented** |
| **FR-06** | **Batch Prediction** | `POST /api/v1/predictions/batch` must process arrays of up to 1,000 records concurrently with bounded latency. | **Implemented** |
| **FR-07** | **Prediction Audit Trail** | `GET /api/v1/predictions/{id}` must retrieve audited prediction logs stored in the relational database. | **Implemented** |
| **FR-08** | **Federated Experiment Registration**| `POST /api/v1/experiments` must validate strategy, client counts, rounds, and adversarial settings, launching background tasks. | **Implemented** |
| **FR-09** | **Experiment Listing & Metrics** | `GET /api/v1/experiments` and `/experiments/{id}/metrics` must report training progress across all rounds. | **Implemented** |
| **FR-10** | **XAI Feature Attribution** | `POST /api/v1/explanations` must compute feature sensitivities, return top positive/negative drivers, and output plain-English incident summaries. | **Implemented** |

---

## 7. Non-Functional Requirements (NFR)

| Identifier | Category | Requirement Specification |
| :--- | :--- | :--- |
| **NFR-01** | **Performance** | Inference latency must remain under 10 ms per record after warm-up; batch inference under 100 ms for 100 records. |
| **NFR-02** | **Reliability** | The application must handle invalid inputs (NaNs, Infs, missing keys, out-of-bound lengths) gracefully with HTTP 422. |
| **NFR-03** | **Security** | Zero arbitrary command execution or raw shell spawning; strict input bounds checking; sensitive stack traces suppressed in 500 errors. |
| **NFR-04** | **Maintainability**| Clear separation between API routes, service business logic, SQLAlchemy ORM repositories, and ML model loading. |
| **NFR-05** | **Portability** | Multi-stage Docker container build runnable locally via Docker Compose or on free-tier container platforms (Render, HF Spaces). |
| **NFR-06** | **Testability** | Test suite must be automated via `pytest` with in-memory SQLite fixtures, achieving 100% pass rate. |

---

## 8. User Workflows

### Workflow 1: Real-Time Flow Triage
```
Security Probe / Sensor
         │
         │ 1. POST /api/v1/predictions (29 flow metrics)
         ▼
FastAPI Backend
         │ 2. Validate input schema (Pydantic v2)
         │ 3. Normalize via cached MinMaxScaler
         │ 4. Execute inference on warm Keras model
         │ 5. Asynchronously persist audit log
         ▼
Response: { predicted_class: "DoS attacks-Hulk", confidence: 0.984, latency_ms: 2.1 }
```

### Workflow 2: Incident Root-Cause Explanation
```
Security Analyst
         │
         │ 1. POST /api/v1/explanations (Suspicious flow vector)
         ▼
Explanation Service
         │ 2. Predict base class probability
         │ 3. Perform local sensitivity perturbation
         │ 4. Rank features and map to global SHAP baseline
         ▼
Response: Top drivers (Dst Port, Protocol, Flow Duration) + SOC Summary text
```

---

## 9. Success Metrics & Acceptance Criteria

1. **Inference Correctness:** Accuracy on CSE-CIC-IDS2018 clean test partition $\ge 90\%$ (FedProx achieved **90.93%**).
2. **API Throughput:** Can serve $\ge 50$ requests per second on standard 2-vCPU hardware.
3. **Zero Security Vulnerabilities:** Clean static scan, zero hardcoded secrets, no unauthenticated shell access.
4. **Test Suite Completeness:** All 19 core test cases pass reliably across clean environments.

---

## 10. Release Milestones & Definition of Done

* **v1.0.0 (Current MVP):** Complete FastAPI backend, SQLAlchemy persistence, ModelManager with weights caching, single/batch prediction, explainability endpoint, 19 passing tests, Docker & Docker Compose.
* **v1.1.0 (Target Free Deployment):** Container image size pruned (<300 MB), GitHub Actions CI pipeline, live deployment to free-tier cloud host.
* **Definition of Done (DoD):** Code implemented, 100% tests passing, documentation reconciled, zero deployment blockers, verified benchmark evidence.
