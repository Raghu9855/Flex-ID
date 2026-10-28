# FLEX-ID: Automated Testing Guide & Strategy

This document details the automated testing framework, test suite architecture, fixture lifecycle, and verification workflows for the **FLEX-ID** backend application.

---

## 1. Testing Philosophy & Test Pyramid

The testing strategy follows industry-standard backend and ML serving testing patterns:

```
          / \
         /   \      E2E / Load Tests (scripts/load_test.py)
        /-----\
       /       \    API & Integration Tests (tests/test_*.py with TestClient)
      /---------\
     /           \  Unit & ML Core Tests (tests/test_inference_service.py)
    /-------------\
```

1. **Unit Tests:** Validate individual functions in isolation (feature extraction, NaN handling, normalization clipping, schema models).
2. **Integration Tests:** Exercise the complete request-response flow from HTTP routers through dependency injection, services, and repository layers down to an isolated in-memory SQLite database.
3. **ML Serving Tests:** Validate that model weights load correctly, inference returns valid probability distributions summing to 1.0, and explanations produce ranked feature contributions.
4. **Performance Tests:** Automated load harness measuring latency distributions (p50, p95, p99) under concurrent load.

---

## 2. Test Architecture & Structure

All tests reside under the `tests/` directory:

```
tests/
├── conftest.py                # Test fixtures: SQLite in-memory engine, TestClient, sample data
├── test_health.py             # Health check and dependency status endpoint tests
├── test_models.py             # Model metadata, version listing, and 404 validation
├── test_predictions.py        # Single and batch inference, payload formats, error handling
├── test_experiments.py       # Benchmark results, experiment history, simulation trigger
├── test_explanations.py       # SHAP explanation and attribution ranking tests
└── test_inference_service.py  # Unit tests for preprocessing, clipping, and feature alignment
```

### Key Fixtures in `conftest.py`

- **`db_session`:** Creates an isolated SQLite in-memory database (`sqlite:///:memory:`), creates all tables from SQLAlchemy models, seeds baseline model records, yields a clean session to each test, and drops tables on teardown.
- **`client`:** Overrides FastAPI's `get_db` dependency to inject the test database session, yielding an instance of `starlette.testclient.TestClient`.
- **`sample_flow_dict`:** Canonical 29-feature dictionary matching CIC-IDS2018 columns.
- **`sample_flow_list`:** 29-element float list representing normalized feature values.

---

## 3. Test Suites Summary

| Test File | Total Tests | Focus Area |
| :--- | :---: | :--- |
| `tests/test_health.py` | 2 | `/health` endpoint status, database connectivity verification |
| `tests/test_models.py` | 4 | Model listing (`/api/v1/models`), individual model details, version not found (404) |
| `tests/test_predictions.py` | 4 | Single prediction with list input, named dict input, batch prediction, invalid feature length (422) |
| `tests/test_experiments.py` | 4 | Experiment listing, experiment comparison retrieval, metrics integrity, trigger simulation |
| `tests/test_explanations.py` | 1 | SHAP explanation endpoint, attribution array validation, top_k ordering |
| `tests/test_inference_service.py` | 4 | Dict feature extraction, list feature extraction, NaN/Inf handling, normalization clipping |
| **Total** | **19** | **100% Passing (0 failures, 0 warnings)** |

---

## 4. Execution Commands

### 4.1 Running All Tests via PowerShell (Windows)
```powershell
# Using the run.ps1 wrapper
.\run.ps1 test

# Or directly with pytest
.\venv\Scripts\pytest.exe -v
```

### 4.2 Running All Tests on Linux / macOS
```bash
pytest -v
```

### 4.3 Running Specific Test Modules
```powershell
# Run only prediction tests
.\venv\Scripts\pytest.exe tests/test_predictions.py -v

# Run only inference service unit tests
.\venv\Scripts\pytest.exe tests/test_inference_service.py -v
```

### 4.4 Running with Coverage Reporting
```powershell
.\venv\Scripts\pytest.exe --cov=app --cov-report=term-missing tests/
```

### 4.5 Running Inside Docker Container
```bash
docker build -t flex-id-backend:test .
docker run --rm flex-id-backend:test pytest -v
```

---

## 5. Defensive Edge Cases Tested

The test suite explicitly asserts defensive behavior against known AI serving edge cases:

1. **Feature Vector Shape Mismatch:** Sending 28 or 30 features instead of 29 returns an immediate HTTP 422 with a structured error explanation rather than crashing inside matrix multiplication.
2. **Unseen Feature Bounds / Outliers:** Raw features exceeding `[0.0, 1.0]` are clipped defensibly by `_preprocess_batch()` to prevent numerical explosion or unexpected model activation behavior.
3. **Missing or Corrupted Model Version:** Requesting a non-existent model version returns HTTP 404 with standard error envelope.
4. **NaN or Infinite Sensor Values:** Preprocessing automatically replaces NaNs with `0.0` and infinite values with bounded values (`posinf=1.0`, `neginf=0.0`).
