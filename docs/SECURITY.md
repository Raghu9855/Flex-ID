# FLEX-ID: Security Architecture & Threat Model

**Document Version:** 1.0.0  
**Status:** Approved  
**Classification:** Public Security Advisory & Architecture Document  

---

## 1. Threat Model & Assets

### 1.1 Protected Assets
1. **Host & Operating System Integrity:** Protection against arbitrary code execution, unauthorized subprocess spawning, and host filesystem tampering.
2. **Model Weights & Artifacts:** Checkpoints (`round-10-weights.pkl`) and scalers (`scaler.pkl`) protected against adversarial tampering.
3. **Database Audit Records:** Prevention of unauthorized deletion or SQL injection into prediction audit logs and experiment metrics.
4. **API Service Availability:** Protection against denial-of-service (DoS) via oversized batch payloads or compute-heavy XAI requests.

### 1.2 Threat Actors & Attack Vectors
* **External Network Attackers:** Attempting to exploit web vulnerabilities (RCE, SQL injection, path traversal, unbounded memory exhaustion).
* **Adversarial Federated Clients:** Malicious edge nodes attempting data poisoning (label flipping) or model replacement (Byzantine update scaling) to degrade intrusion detection capabilities.
* **Malicious Network Traffic (Evasion Attacks):** Crafting packet statistics designed to bypass detection thresholds.

---

## 2. Security Vulnerability Audit & Remediation (SEC-01)

### The Legacy Node.js Remote Code Execution (RCE) Flaw

#### Vulnerable Code Identified in `backend/server.js`:
```javascript
// Lines 77-80 & 156-173 in backend/server.js
function spawnPythonProcess(command, args, processId, socket) {
    const proc = spawn(command, finalArgs, {
        cwd: PROJECT_ROOT,
        shell: true // <-- CRITICAL: Vulnerable to command chaining and injection!
    });
}

app.post('/api/run-command', (req, res) => {
    const { command, args, id } = req.body;
    spawnPythonProcess(command, args || [], id, io);
});
```

#### Vulnerability Assessment:
- **CVSS 3.1 Score:** **9.8 (Critical)** — `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`
- **Root Cause:** The endpoint accepted arbitrary client-provided commands (`command`) and executed them with `shell: true`. An unauthenticated HTTP request could pass `; rm -rf /` or PowerShell command strings, giving complete remote control of the server.
- **Remediation:**
  1. Deprecated the legacy Node.js server.
  2. Built the **FastAPI backend** in Python with zero arbitrary shell execution endpoints.
  3. All system operations are strictly programmatic function calls within Python.
  4. Docker configurations and launcher scripts now target `uvicorn app.main:app` exclusively.

---

## 3. Defense-in-Depth Security Controls

### 3.1 Input Validation & Strict Schemas
- Implemented with **Pydantic v2** in [app/schemas/](file:///c:/Users/HP/Flex-ID/app/schemas/):
  - Every flow feature is strictly validated as float.
  - Required canonical feature list length is enforced (must equal 29; otherwise rejected with HTTP 422).
  - Feature counts and array lengths are bounded. Batch requests are capped at `MAX_BATCH_SIZE = 1000` (rejecting excessive payloads with HTTP 400).
  - Preprocessing uses `np.nan_to_num` and `np.clip` to neutralize $\pm \infty$ and NaN injection attempts.

### 3.2 SQL Injection Prevention
- All database operations utilize **SQLAlchemy 2.0 ORM** parameterized queries (`select()`, `where()`).
- Zero dynamic string concatenation or raw SQL text queries in application routes.

### 3.3 Path Traversal Prevention
- All filesystem paths in [app/core/config.py](file:///c:/Users/HP/Flex-ID/app/core/config.py) and [app/services/model_manager.py](file:///c:/Users/HP/Flex-ID/app/services/model_manager.py) resolve against an explicit, pinned `PROJECT_ROOT` path.
- API endpoints do not accept arbitrary filesystem paths; users reference registered model versions (e.g. `fedprox-v1`), not file paths.

### 3.4 Deserialization Security (`pickle`)
- **Risk:** Python's `pickle` library can execute arbitrary code during unpickling if reading untrusted inputs.
- **Control:** The application loads pickle files **only from local, static project directories** (`data/` and `results/`).
- **Policy:** The API does not accept uploaded pickle files from HTTP callers.

### 3.5 Sensitive Data Suppression & Error Envelopes
- Raw database exceptions or Python tracebacks are caught by `global_exception_handler` in [app/core/exceptions.py](file:///c:/Users/HP/Flex-ID/app/core/exceptions.py).
- Callers receive a sanitized JSON error envelope (`{"error": "InternalServerError", "message": "An unexpected error occurred..."}`) without leaking directory paths, database connection strings, or system credentials.

---

## 4. Container & Operating System Security

The production [Dockerfile](file:///c:/Users/HP/Flex-ID/Dockerfile) implements industry best practices:
1. **Non-Root Execution:**
   ```dockerfile
   RUN groupadd -g 1001 appgroup && \
       useradd -u 1001 -g appgroup -m -s /bin/bash appuser
   USER appuser
   ```
   The process runs as an unprivileged user (UID 1001), preventing container breakout exploits from obtaining root privileges on the host.
2. **Minimal Surface Area:**
   - Multi-stage build discards compilation tools (`gcc`, `build-essential`) in the final runtime stage.
   - Package cache is purged (`rm -rf /var/lib/apt/lists/*`).
3. **Container Health Probes:**
   - Employs a non-intrusive `HEALTHCHECK` probe targeting `http://localhost:8000/health`.

---

## 5. Federated Learning Poisoning Defenses

Adversarial clients can attempt to degrade model performance. FLEX-ID implements and evaluates algorithmic defenses:
* **Non-IID Drift vs. Adversarial Attacks:** FedProx protects against non-IID client drift through proximal regularization ($\frac{\mu}{2}\|w - w_t\|^2$), maintaining **88.82% accuracy** under label flipping.
* **Byzantine Robustness:** For hostile model poisoning (e.g., Byzantine update amplification), [ml/aggregation.py](file:///c:/Users/HP/Flex-ID/ml/aggregation.py) provides **Krum**, **Multi-Krum**, **Trimmed Mean**, and **Coordinate Median** to filter out anomalous client update vectors.

---

## 6. Residual Risks & Future Hardening

1. **API Authentication:** The current MVP operates without an API gateway / token layer. For public internet exposure, reverse-proxy authentication (e.g. JWT or API-key bearer tokens) must be placed in front of endpoints.
2. **Rate Limiting:** Network sensors could flood the `/predictions` endpoint. Adding Redis-backed rate limiting (e.g., `slowapi`) is recommended for production edge deployment.
