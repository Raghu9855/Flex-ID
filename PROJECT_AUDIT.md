# FLEX-ID Technical Project Audit

**Auditor:** Senior Backend Engineer & AI Systems Architect  
**Project:** FLEX-ID (Federated Learning Explainable Intrusion Detection System)  
**Repository:** `c:\Users\HP\Flex-ID`  
**Date:** October 2026  
**Status:** Phase 1 Complete (Audit & Gap Analysis)

---

## 1. Executive Summary & Reality Check

FLEX-ID was conceived as a federated intrusion detection research prototype demonstrating privacy-preserving training on the CSE-CIC-IDS2018 dataset, featuring FedAvg and FedProx, 5 adversarial attack simulations, Byzantine-robust aggregations (Krum, Multi-Krum, Trimmed Mean, Coordinate Median), and federated SHAP explanations.

A comprehensive technical audit of the codebase reveals major discrepancies between previous assumptions/claims and the physical codebase, along with critical architectural, security, and machine learning engineering gaps:

1. **Framework Mismatch (PyTorch vs. TensorFlow/Keras):**
   * **Claim:** The project uses PyTorch.
   * **Reality:** The codebase exclusively uses **TensorFlow / Keras** (`ml/model.py`, `ml/client.py`, `ml/client_attack.py`, `ml/compare_results.py`). All models are built with `keras.Sequential`, trained using `tf.GradientTape` (FedProx) and `model.fit()`, and saved as pickled NumPy weight arrays.
2. **Backend Architecture & Critical Security Flaw:**
   * **Claim:** Backend-oriented AI system.
   * **Reality:** The existing backend is a **Node.js / Express** server (`backend/server.js`) that spawns Python scripts as unmanaged child processes via CLI.
   * **Critical Remote Code Execution (RCE) Vulnerability:** In `backend/server.js`, `POST /api/run-command` accepts arbitrary shell strings `{ command, args }` and executes them via `child_process.spawn(..., { shell: true })`. This allows arbitrary remote command execution on the host machine.
   * **Zero REST Inference Endpoints:** There is no `POST /predict` endpoint, no real-time inference pipeline, and no model serving infrastructure.
3. **Machine Learning Pipeline Integrity (The "Missing Scaler" Bug):**
   * In `ml/1_process_data.py`, `MinMaxScaler()` is fitted on the raw dataset and used to scale features, but the fitted scaler is **never saved to disk** (no `scaler.pkl`).
   * Consequently, raw inference data cannot be deterministically normalized to match training distribution parameters (`min_`, `scale_`), leading to severe feature distribution skew and erratic predictions during live inference.
4. **Discrepancy in Performance Claims:**
   * **Preliminary Claims:** Clean-data accuracy: FedProx 92.13%, FedAvg 89.32%; under label-flipping attack: FedProx 88.88%, FedAvg 73.73%.
   * **Verified Recorded Artifacts (`results/comparison_results_*.json`, `results/metrics_summary.json` at Round 10):**
     * **No Attack (Clean):** FedProx Accuracy **90.93%** (Macro F1: **0.5899**, Weighted F1: **0.9109**); FedAvg Accuracy **81.47%** (Macro F1: **0.4350**, Weighted F1: **0.8382**).
     * **Under Attack (Label Flipping):** FedProx Accuracy **88.82%** (Macro F1: **0.5186**, Weighted F1: **0.8495**); FedAvg Accuracy **85.49%** (Macro F1: **0.4422**, Weighted F1: **0.8577**).
     * High accuracy numbers are heavily dominated by the majority Benign class (~73.7% of test samples). Macro F1 reveals that minority attack detection remains a significant challenge, which must be honestly represented.
5. **Zero Persistence & Zero Tests:**
   * No database exists (no SQLite, PostgreSQL, or MySQL). All states and metrics are stored in flat JSON, PNG, or PKL files.
   * No automated test suite exists (`tests/` directory is missing).

---

## 2. Detailed Component Audit

### 2.1 Project Structure & Separation of Concerns
* **Current State:**
  * Root contains mixed scripts (`run.ps1`, `Dockerfile`, `deployment_guide.md`).
  * `ml/` contains sequential numbered scripts (`1_process_data.py`, `2_create_partitions.py`, `4_server.py`) intermingled with reusable logic (`model.py`, `aggregation.py`).
  * `backend/` is Node.js rather than Python.
  * `frontend/` is a Vite + React application expecting socket-based log streaming and specific endpoints.
* **Problems:** Tight coupling to local working directories; scripts rely on relative path assumptions (`data/` and `results/`). Spawning Python scripts via Node.js causes process lifecycle leaks and platform-dependent `taskkill` commands.

### 2.2 Model Architecture & Inference Pipeline
* **File:** `ml/model.py`
  * Architecture: 3-layer MLP (`Dense(256) -> BatchNorm -> Dropout(0.3) -> Dense(128) -> BatchNorm -> Dropout(0.3) -> Dense(64) -> BatchNorm -> Dropout(0.3) -> Dense(num_classes, softmax)`).
  * Loss: `sparse_categorical_crossentropy`. Optimizer: `Adam`.
* **Serialization:** Models are not saved as standard Keras models (`.keras` or `.h5`). Instead, `save_parameters()` in `4_server.py` serializes a list of NumPy weight arrays (`round-*-weights.pkl`).
* **Inference Gap:** To run inference, a consumer must manually inspect the feature count, load `label_encoder.pkl`, instantiate `create_dnn_model(x_dim, num_classes)`, call `model.set_weights(pickle.load(...))`, and invoke `model.predict()`. There is no encapsulation or caching.

### 2.3 Data Preprocessing & Partitioning
* **Files:** `ml/1_process_data.py`, `ml/2_create_partitions.py`
* **Features:** 29 curated features extracted from 80 CIC-IDS2018 columns:
  `Dst Port`, `Protocol`, `Flow Duration`, `Tot Fwd Pkts`, `Tot Bwd Pkts`, `TotLen Fwd Pkts`, `TotLen Bwd Pkts`, `Fwd Pkt Len Max`, `Fwd Pkt Len Mean`, `Bwd Pkt Len Max`, `Bwd Pkt Len Mean`, `Flow Byts/s`, `Flow Pkts/s`, `Flow IAT Mean`, `Flow IAT Max`, `Fwd IAT Mean`, `Bwd IAT Mean`, `Fwd Header Len`, `Bwd Header Len`, `Fwd Pkts/s`, `Bwd Pkts/s`, `Pkt Len Mean`, `Pkt Len Max`, `Pkt Len Var`, `SYN Flag Cnt`, `RST Flag Cnt`, `ACK Flag Cnt`, `Init Fwd Win Byts`, `Init Bwd Win Byts`.
* **Preprocessing Defect:** `MinMaxScaler` is fit and applied in-memory, but the fitted instance is never exported. Without `scaler.pkl`, production inference is impossible because incoming network packet flows cannot be scaled with identical statistics.
* **Partitioning:** Dirichlet-style non-IID partition assigns Client 0 a disproportionate share of attack traffic (`alpha = 0.4`), while benign traffic is distributed evenly. Handled cleanly in `2_create_partitions.py`.

### 2.4 Federated Learning Workflow (Server & Clients)
* **Files:** `ml/4_server.py`, `ml/client.py`, `ml/client_attack.py`, `ml/aggregation.py`
* **Aggregation:** Cleanly written implementations of Krum, Multi-Krum, Trimmed Mean, and Coordinate Median in `aggregation.py`.
* **FedProx:** Implemented properly on the client via `tf.GradientTape` by adding proximal penalty $\frac{\mu}{2} \sum \|w_{local} - w_{global}\|^2$ to loss.
* **Flower Coupling:** Flower (`flwr`) requires running individual client processes that connect over gRPC to port 8080. It is a simulation running on a single machine, not a distributed production deployment across physical nodes.
* **Port Handling:** `force_release_port` in `4_server.py` executes shell commands (`netstat`, `taskkill /F`) to kill whatever is listening on port 8080. This is brittle, non-portable, and dangerous in shared hosting environments.

### 2.5 Adversarial Attacks
* **File:** `ml/client_attack.py`
* Supports 5 attack types:
  1. `flip`: Relabels malicious flows as Benign (data poisoning).
  2. `noise`: Injects Gaussian noise $\mathcal{N}(0, \sigma^2)$ into weight updates.
  3. `backdoor`: Stamps a backdoor trigger value onto a feature and sets label to Benign.
  4. `byzantine`: Multiplies the weight update delta by a scale factor.
  5. `adaptive`: Executes gradient ascent on cross-entropy loss followed by L2 norm clipping.
* **Findings:** Attack implementations are well-structured. However, in `client.py` and `client_attack.py`, SMOTE and oversampling are applied inside the client script, consuming significant CPU during initialization.

### 2.6 Explainable AI (SHAP)
* **File:** `ml/explain_model.py`
* **Mechanism:** Uses `shap.KernelExplainer` with 100 background samples and 50 explained test samples per client. Computes mean absolute SHAP values and computes Kendall's Tau and Spearman's rank correlation across clients.
* **Latency & Bottleneck:** `shap.KernelExplainer` takes between 30 to 120 seconds to explain 50 samples across 13 classes. If called synchronously inside a web endpoint, it will trigger HTTP gateway timeouts.
* **Privacy Claim:** The script correctly computes SHAP locally and averages feature rankings before outputting `shap_summary.json`.

### 2.7 Existing Backend & Security Vulnerabilities
* **File:** `backend/server.js`
* **RCE Vulnerability:**
  ```javascript
  app.post('/api/run-command', (req, res) => {
      const { command, args, id } = req.body;
      spawnPythonProcess(command, args || [], id, io); // shell: true in spawnPythonProcess!
  });
  ```
  This is a critical security vulnerability allowing unrestricted command execution.
* **Unvalidated File Uploads:** `app.post('/api/upload-dataset')` writes uploaded files directly to `data/combined_ids2018_raw.csv` without file-type validation, size limits, or content verification.
* **Absence of Core Backend Capabilities:** No authentication, no request schema validation, no database, no caching, and no error standardization.

### 2.8 Existing Frontend
* **Directory:** `frontend/` (React + Vite + TailwindCSS).
* Designed as an operations dashboard: monitors processes, views logs via Socket.IO, plots confusion matrices, visualizes network topology, and browses files.
* Can be maintained or bridged to the new FastAPI backend by providing compatible status and metadata endpoints.

---

## 3. Prioritized Classification of Issues

| Priority | Category | Issue Description | Recommended Solution |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | Security | Arbitrary Command Execution in `server.js` (`POST /api/run-command`). | Replace Node.js server with FastAPI backend; never expose raw shell execution endpoints. |
| **CRITICAL** | ML Engineering | Preprocessing transformer `MinMaxScaler` is not saved (`scaler.pkl`). | Fit and serialize `scaler.pkl` in preprocessing; load it in the inference service for consistent scaling. |
| **CRITICAL** | Architecture | Complete lack of ML serving and inference endpoints. | Implement `POST /api/v1/predictions` and `POST /api/v1/predictions/batch` in FastAPI with Pydantic validation. |
| **HIGH** | Reliability | Model loading is un-encapsulated; weights are pickled NumPy arrays. | Build a robust `ModelManager` / `InferenceService` with in-memory model caching and warm startup. |
| **HIGH** | Architecture | Zero database persistence; results, logs, and experiments are ephemeral files. | Integrate SQLAlchemy ORM with SQLite (dev/test) and MySQL (production) for experiments, models, and predictions. |
| **HIGH** | Performance | SHAP `KernelExplainer` is too computationally expensive for real-time API requests. | Design an asynchronous or cached explanation service with fast sample attribution and precomputed global baselines. |
| **MEDIUM** | Documentation | Inaccurate claim that the project uses PyTorch and overstated preliminary metrics. | Correct documentation to accurately reflect TensorFlow/Keras, report verified F1/Accuracy metrics, and update resume bullets. |
| **MEDIUM** | Deployment | Dockerfile uses Python 3.9 while local codebase runs Python 3.13; lacks health checks. | Modernize Dockerfile to multi-stage Python 3.11/3.13 build, non-root user, proper health checks, and docker-compose. |
| **MEDIUM** | Orchestration | Synchronous FL training execution blocks threads; uses aggressive `taskkill`. | Run federated training as an asynchronous background worker with state tracking. |
| **LOW** | Code Cleanliness | Duplicated preprocessing snippets across `client.py` and `client_attack.py`. | Refactor common data-loading and SMOTE logic into `ml/utils/data_utils.py`. |
| **LOW** | Frontend | Frontend tightly coupled to Socket.IO process logs. | Provide lightweight WebSocket or Server-Sent Events (SSE) log streaming in FastAPI if frontend log viewing is retained. |

---

## 4. Verified Empirical Results

All evaluations conducted on CSE-CIC-IDS2018 test partition (144,051 samples, 13 classes, seed = 42, 10 federated rounds):

| Metric | FedAvg (Clean) | FedProx (Clean) | FedAvg (Label Flip Attack) | FedProx (Label Flip Attack) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 81.47% | **90.93%** | 85.49% | **88.82%** |
| **Balanced Accuracy** | 60.38% | **74.01%** | 56.67% | **63.60%** |
| **Macro F1-Score** | 0.4350 | **0.5899** | 0.4422 | **0.5186** |
| **Weighted F1-Score** | 0.8382 | **0.9109** | 0.8577 | **0.8495** |
| **ROC-AUC (Macro)** | 0.9496 | **0.9812** | 0.9792 | **0.9731** |
| **PR-AUC (Macro)** | 0.5767 | **0.6510** | 0.6348 | **0.6391** |
| **Matthews Corr. (MCC)**| 0.6275 | **0.8074** | 0.6809 | **0.7295** |
| **Inference Time (ms)**| ~5,840 ms | ~5,356 ms | ~5,862 ms | ~5,523 ms |

### Key Analytical Takeaways:
1. **FedProx Superiority:** Under non-IID conditions ($\alpha = 0.4$), FedProx significantly outperforms FedAvg in clean data (+9.46% Accuracy, +0.155 Macro F1, +0.18 MCC).
2. **Attacks & Robustness:** Under label flipping, FedProx maintains higher Macro F1 (0.5186 vs 0.4422) and MCC (0.7295 vs 0.6809), showing greater stability against gradient deviation caused by poisoned clients.
3. **Class Imbalance Impact:** Because Benign traffic comprises 73.7% of the dataset, overall accuracy is an incomplete metric. Macro F1 and MCC must be emphasized in resumes and documentation to reflect genuine engineering rigor.
