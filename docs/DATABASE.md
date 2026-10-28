# FLEX-ID: Database Architecture & Relational Persistence

**Document Version:** 1.0.0  
**ORM:** SQLAlchemy 2.0  
**Supported Dialects:** SQLite 3.x (Development / CI / Test), MySQL 8.0 (Production)  

---

## 1. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    EXPERIMENTS ||--o{ EXPERIMENT_METRICS : "tracks rounds"
    EXPERIMENTS ||--o{ MODEL_RECORDS : "generates checkpoints"
    MODEL_RECORDS ||--o{ PREDICTION_LOGS : "serves inferences"

    EXPERIMENTS {
        string id PK "UUID"
        string name "Human-readable label"
        string strategy "fedavg | fedprox | krum | median"
        integer rounds "Total training rounds"
        integer num_clients "Participating logical clients"
        float proximal_mu "FedProx regularizer (default 0.1)"
        boolean is_attack "Adversarial simulation flag"
        string attack_type "flip | noise | backdoor | byzantine"
        string status "running | completed | failed"
        text error_message "Failure stack trace"
        datetime created_at "UTC timestamp"
        datetime completed_at "UTC timestamp"
    }

    EXPERIMENT_METRICS {
        integer id PK "Auto-increment"
        string experiment_id FK "References experiments.id"
        integer round_number "Round sequence (1..N)"
        float train_loss "Mean client training loss"
        float accuracy "Distributed test accuracy"
        float f1_score "Distributed macro/weighted F1"
        datetime created_at "UTC timestamp"
    }

    MODEL_RECORDS {
        string id PK "UUID"
        string model_version UK "e.g., fedprox-v1 (Indexed)"
        string strategy "Aggregation strategy"
        string weights_path "Relative path to weights pickle"
        string experiment_id FK "References experiments.id"
        integer input_features_count "Default 29"
        integer num_classes "Default 13"
        float accuracy "Baseline test accuracy"
        float macro_f1 "Baseline macro F1"
        float weighted_f1 "Baseline weighted F1"
        boolean is_active "Serving traffic flag"
        datetime created_at "UTC timestamp"
    }

    PREDICTION_LOGS {
        string id PK "UUID (prediction_id)"
        string model_id FK "References model_records.id"
        string predicted_class "e.g., Benign, DoS attacks-Hulk"
        integer predicted_class_id "Numeric label"
        float confidence "Softmax score (0.0 - 1.0)"
        float latency_ms "Execution time in ms"
        text input_summary "Statistical feature summary"
        datetime created_at "UTC timestamp (Indexed)"
    }
```

---

## 2. Table Specifications & Indexes

### 2.1 `experiments`
| Column | Type | Nullable | Constraints & Defaults | Notes |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `VARCHAR(36)` | No | **PK** (UUID v4) | Unique experiment identifier |
| `name` | `VARCHAR(128)`| No | None | Human-readable experiment description |
| `strategy` | `VARCHAR(64)` | No | Default: `'fedprox'` | `fedavg`, `fedprox`, `krum`, `median` |
| `rounds` | `INTEGER` | No | Default: `10` | Total rounds |
| `num_clients` | `INTEGER` | No | Default: `4` | Number of logical clients |
| `proximal_mu` | `FLOAT` | No | Default: `0.1` | FedProx proximal coefficient |
| `is_attack` | `BOOLEAN` | No | Default: `False` | True if adversarial client is present |
| `attack_type` | `VARCHAR(64)` | Yes | Default: `NULL` | Attack method if applicable |
| `status` | `VARCHAR(32)` | No | Default: `'completed'` | `'running'`, `'completed'`, `'failed'` |
| `error_message` | `TEXT` | Yes | Default: `NULL` | Captured exception trace if failed |
| `created_at` | `DATETIME` | No | Default: `UTC NOW` | Audit creation timestamp |
| `completed_at` | `DATETIME` | Yes | Default: `NULL` | Experiment completion timestamp |

### 2.2 `experiment_metrics`
| Column | Type | Nullable | Constraints | Notes |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INTEGER` | No | **PK** (Auto-increment) | Metric row ID |
| `experiment_id` | `VARCHAR(36)`| No | **FK** -> `experiments.id` | Cascades on delete (`ondelete='CASCADE'`) |
| `round_number` | `INTEGER` | No | None | FL round index |
| `train_loss` | `FLOAT` | Yes | None | Weighted mean client train loss |
| `accuracy` | `FLOAT` | Yes | None | Distributed evaluation accuracy |
| `f1_score` | `FLOAT` | Yes | None | Distributed evaluation F1 |
| `created_at` | `DATETIME` | No | Default: `UTC NOW` | Record timestamp |
* **Composite Index:** `ix_metric_exp_round` on `(experiment_id, round_number)` for high-speed timeseries plotting.

### 2.3 `model_records`
| Column | Type | Nullable | Constraints | Notes |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `VARCHAR(36)` | No | **PK** (UUID v4) | Model checkpoint ID |
| `model_version` | `VARCHAR(64)` | No | **Unique, Index** | Lookup key (e.g. `fedprox-v1`) |
| `strategy` | `VARCHAR(64)` | No | None | Training strategy |
| `weights_path` | `VARCHAR(255)`| No | None | Local filesystem path to weights `.pkl` |
| `experiment_id` | `VARCHAR(36)` | Yes | **FK** -> `experiments.id` | Source training experiment |
| `input_features_count` | `INTEGER` | No | Default: `29` | Feature vector dimensionality |
| `num_classes` | `INTEGER` | No | Default: `13` | Output classification classes |
| `accuracy` | `FLOAT` | Yes | None | Evaluated test accuracy |
| `macro_f1` | `FLOAT` | Yes | None | Evaluated macro F1-score |
| `weighted_f1`| `FLOAT` | Yes | None | Evaluated weighted F1-score |
| `is_active` | `BOOLEAN` | No | Default: `False` | Active live serving model |
| `created_at` | `DATETIME` | No | Default: `UTC NOW` | Checkpoint timestamp |

### 2.4 `prediction_logs`
| Column | Type | Nullable | Constraints | Notes |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `VARCHAR(36)` | No | **PK** (UUID v4) | Prediction UUID returned to caller |
| `model_id` | `VARCHAR(36)` | No | **FK** -> `model_records.id` | Model checkpoint used for inference |
| `predicted_class`| `VARCHAR(64)`| No | None | Intrusion class label string |
| `predicted_class_id` | `INTEGER` | No | None | Integer index (0..12) |
| `confidence` | `FLOAT` | No | None | Softmax confidence |
| `latency_ms` | `FLOAT` | No | None | Processing latency in ms |
| `input_summary` | `TEXT` | Yes | None | Feature envelope summary |
| `created_at` | `DATETIME` | No | **Index**, Default: `UTC NOW`| Queryable audit timestamp |

---

## 3. Session Lifecycle & Dependency Injection

The application adheres to FastAPI's dependency injection model via `get_db()` in [app/db/session.py](file:///c:/Users/HP/Flex-ID/app/db/session.py):

```python
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- **Scoped Sessions:** Each incoming HTTP request receives a dedicated SQLAlchemy `Session`.
- **Automatic Resource Cleanup:** `finally: db.close()` ensures zero database connection leaks, returning the connection to the connection pool (`pool_pre_ping=True`).
- **Transaction Boundaries:** Repositories commit explicit transactions upon write (`db.commit()`), while read operations are read-only.

---

## 4. SQLite vs. MySQL Dialect Differences

| Feature | SQLite 3 (`development` / `testing`) | MySQL 8.0 (`production`) |
| :--- | :--- | :--- |
| **Engine URL** | `sqlite:///./flex_id.db` | `mysql+pymysql://user:pass@host:3306/db` |
| **Concurrency** | File-level locking (`check_same_thread=False`) | Multi-client row-level locking (InnoDB) |
| **Connection Pooling** | `StaticPool` (in tests) | Managed pool (size=10, overflow=20) |
| **Foreign Keys** | Enabled via connection pragma | Enforced natively by InnoDB engine |
| **Date Storage** | ISO8601 strings | Native `DATETIME(6)` |

---

## 5. Test Database Strategy

All unit and integration tests run against an **in-memory SQLite database (`sqlite:///:memory:`)** managed via [tests/conftest.py](file:///c:/Users/HP/Flex-ID/tests/conftest.py):
- Fast execution: Zero disk I/O, entire 19-test suite completes in $< 3$ seconds.
- Total Isolation: The schema is dropped and re-created in memory for each test session, preventing cross-test pollution.
- Zero External Dependencies: Pytest executes cleanly on developer workstations and GitHub Actions runners without requiring a running database server.
