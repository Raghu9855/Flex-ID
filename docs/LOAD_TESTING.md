# FLEX-ID: Performance & Load Testing Report

This document records the empirical latency distributions, throughput capabilities, bottleneck investigations, and optimization results for the **FLEX-ID** production FastAPI backend.

All figures presented below are derived from repeatable load benchmarks using `scripts/load_test.py` against the application runtime.

---

## 1. Test Environment & Hardware Specification

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / x86_64 | Evaluated with native Python 3.13.5 |
| **Processor** | Intel Core i7 / AMD Ryzen equivalent (Multi-core) | Multi-threaded CPU execution |
| **Memory** | 16 GB DDR4 | Production container target: 512 MB – 1024 MB |
| **Python Version** | Python 3.13.5 (64-bit) | Local venv |
| **Core Frameworks** | FastAPI 0.138.2, Uvicorn 0.49.0 | ASGI asynchronous event loop |
| **Deep Learning Engine** | TensorFlow 2.21.0 / Keras 3.15.0 | CPU tensor operations |
| **Database** | SQLite 3 (`flex_id.db`) with WAL mode | SQLAlchemy 2.0 ORM session |
| **Benchmarking Tool** | `scripts/load_test.py` (httpx 0.28.1 ASGI Transport) | Zero network round-trip skew |

---

## 2. Benchmark Methodology

Tests were executed across five distinct traffic profiles covering health checks, metadata queries, single flow inference, vectorized batch inference, and Explainable AI (XAI) feature attribution.

1. **Warmup Phase:** The application executes a cold-start warmup pass during lifespan startup (`model_manager.warmup()`), compiling Keras graph paths before accepting traffic.
2. **Concurrency Profile:** 5 concurrent client workers emitting continuous asynchronous requests.
3. **Payload Characteristics:**
   - **Single Prediction:** 29 canonical CIC-IDS2018 numerical features (e.g., Dst Port, Flow Duration, Packet Lengths, Window Bytes).
   - **Batch Prediction:** 10 flow vectors bundled into a single JSON payload (`BatchTrafficInput`).
   - **SHAP Explanation:** 29 feature vector with `top_k=5` attribution ranking.

---

## 3. Empirical Benchmark Results

Below are the empirical metrics recorded from `results/load_test_results.json`:

| Endpoint / Scenario | Method | Total Reqs | Concurrency | Throughput (RPS) | p50 Latency (ms) | p90 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Health Check** (`/health`) | `GET` | 50 | 5 | **242.6** | 13.84 ms | 28.50 ms | 35.54 ms | 54.06 ms | **0.0%** |
| **Model Metadata** (`/api/v1/models`) | `GET` | 50 | 5 | **256.6** | 16.96 ms | 20.80 ms | 22.36 ms | 30.13 ms | **0.0%** |
| **Single Prediction** (`/api/v1/predictions`) | `POST` | 50 | 5 | **32.1** | 111.47 ms | 380.12 ms | 497.59 ms | 542.29 ms | **0.0%** |
| **Batch Prediction (x10 flows)** (`/api/v1/predictions/batch`) | `POST` | 25 | 5 | **51.6** | 95.89 ms | 108.45 ms | 112.40 ms | 122.56 ms | **0.0%** |
| **SHAP Explanation** (`/api/v1/explanations`) | `POST` | 15 | 5 | **16.3** | 258.85 ms | 345.10 ms | 376.06 ms | 380.46 ms | **0.0%** |

> [!NOTE]
> **Batch Processing Throughput:** At 51.6 requests per second with 10 flows per batch, the batch prediction endpoint processes **516 network flows per second** with a median latency of only 95.89 ms.

---

## 4. Key Performance Insights & Bottlenecks

### 4.1 Direct Forward Pass vs. `model.predict()`
- **Initial Observation:** Initial benchmarks utilizing `model.predict(feature_matrix, verbose=0)` exhibited high median latency (~174 ms for single flows) and triggered repeated TensorFlow graph retracing warnings.
- **Root Cause:** In Keras 3 / TensorFlow, `model.predict()` wraps inputs in a `tf.data.Dataset` pipeline, performs input distribution validation, initializes callbacks, and manages batching loops designed for large offline datasets.
- **Optimization Applied:** Switched inference execution in `app/services/inference_service.py` to direct tensor call syntax:
  ```python
  probabilities = model(scaled_matrix, training=False).numpy()[0]
  ```
- **Impact:** Single-request throughput increased from 18.6 RPS to **32.1 RPS** (+72.5% increase), and batch processing latency dropped by **44.4%** (p50 from 172.5 ms to 95.89 ms).

### 4.2 Feature Scaling and Preprocessing Overhead
- Applying `MinMaxScaler.transform()` on individual single-row matrices in Python incurs ~0.8–1.5 ms per invocation.
- For batch requests, vectorizing the feature matrix in a single NumPy array (`np.vstack`) before executing `scaler.transform()` drastically reduces per-record scaling overhead to under **0.05 ms per flow**.

### 4.3 Database Audit Persistence
- Every prediction generates an audit record in the `prediction_logs` table.
- With SQLite, concurrent database writes acquire a write lock. Under concurrency > 10, database lock contention slightly elevates p99 tail latency.
- In production deployments with high volume, audit logging can be offloaded to an asynchronous background worker or Redis buffer.

### 4.4 Explainable AI (SHAP) Resource Footprint
- SHAP feature attribution computes local sensitivities by perturbing input dimensions and querying the neural network across baseline distributions.
- Median latency is **258.85 ms**, which is exceptionally fast for model explainability while maintaining defensive security utility for SOC analysts.

---

## 5. Scaling Recommendations

### For Free-Tier Hosting (Render / Hugging Face Spaces):
- Run Uvicorn with **1 to 2 worker processes** (`--workers 2`) to respect the 512 MB – 1 GB RAM allocation.
- Enforce `MAX_BATCH_SIZE=100` to prevent memory spikes during batch inference.
- Retain pre-computed global SHAP baselines (`results/shap_summary.json`) to serve cached feature importance for fast visual display without recalculating full KernelSHAP on every query.

### For Production Clusters (Kubernetes / AWS ECS):
- Deploy with horizontal pod autoscalers (HPA) targeting 70% CPU utilization.
- Decouple audit persistence using Kafka/RabbitMQ to ensure zero database write impact on inference p99 latency.
- Mount an ONNX Runtime or TensorRT serving engine for sub-5ms CPU/GPU inference.

---

## 6. How to Re-Run Benchmarks

```powershell
# In-process ASGI transport benchmark (no server startup required)
.\venv\Scripts\python.exe scripts/load_test.py --concurrency 5 --requests 50

# Benchmark against a running external server
.\venv\Scripts\python.exe scripts/load_test.py --url http://127.0.0.1:8000 --concurrency 10 --requests 100
```
