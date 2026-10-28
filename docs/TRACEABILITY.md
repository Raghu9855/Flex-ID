# FLEX-ID: Requirements Traceability Matrix (RTM)

**Document Version:** 1.0.0  
**Status:** Approved  
**Coverage:** 100% of Functional & Non-Functional Requirements Mapped to Tests  

---

## 1. Traceability Matrix

| PRD Req | SRS Section | Implementation Module | API Route / Interface | Automated Test | Verification Status |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **FR-01** | §3.1 | [app/api/v1/health.py](file:///c:/Users/HP/Flex-ID/app/api/v1/health.py) | `GET /health`<br>`GET /api/v1/health` | `tests/test_health.py::test_root_health_check`<br>`tests/test_health.py::test_api_v1_health_check` | **VERIFIED (PASSED)** |
| **FR-02** | §3.2 | [app/api/v1/models.py](file:///c:/Users/HP/Flex-ID/app/api/v1/models.py)<br>[app/repositories/model_repository.py](file:///c:/Users/HP/Flex-ID/app/repositories/model_repository.py) | `GET /api/v1/models` | `tests/test_models.py::test_list_models` | **VERIFIED (PASSED)** |
| **FR-03** | §3.3 | [app/api/v1/models.py](file:///c:/Users/HP/Flex-ID/app/api/v1/models.py) | `GET /api/v1/models/{version}` | `tests/test_models.py::test_get_model_details`<br>`tests/test_models.py::test_get_nonexistent_model` | **VERIFIED (PASSED)** |
| **FR-04** | §3.4 | [app/api/v1/models.py](file:///c:/Users/HP/Flex-ID/app/api/v1/models.py)<br>[app/services/model_manager.py](file:///c:/Users/HP/Flex-ID/app/services/model_manager.py) | `POST /api/v1/models/{version}/activate` | `tests/test_models.py::test_activate_model` | **VERIFIED (PASSED)** |
| **FR-05** | §3.5 | [app/services/inference_service.py](file:///c:/Users/HP/Flex-ID/app/services/inference_service.py)<br>[app/api/v1/predictions.py](file:///c:/Users/HP/Flex-ID/app/api/v1/predictions.py) | `POST /api/v1/predictions` | `tests/test_predictions.py::test_single_prediction_dict_input`<br>`tests/test_predictions.py::test_single_prediction_list_input` | **VERIFIED (PASSED)** |
| **FR-06** | §3.6 | [app/services/inference_service.py](file:///c:/Users/HP/Flex-ID/app/services/inference_service.py)<br>[app/api/v1/predictions.py](file:///c:/Users/HP/Flex-ID/app/api/v1/predictions.py) | `POST /api/v1/predictions/batch` | `tests/test_predictions.py::test_batch_prediction` | **VERIFIED (PASSED)** |
| **FR-07** | §3.7 | [app/repositories/prediction_repository.py](file:///c:/Users/HP/Flex-ID/app/repositories/prediction_repository.py) | `GET /api/v1/predictions/{id}` | `tests/test_predictions.py::test_single_prediction_dict_input` | **VERIFIED (PASSED)** |
| **FR-08** | §3.8 | [app/services/fl_service.py](file:///c:/Users/HP/Flex-ID/app/services/fl_service.py)<br>[app/api/v1/experiments.py](file:///c:/Users/HP/Flex-ID/app/api/v1/experiments.py) | `POST /api/v1/experiments` | `tests/test_experiments.py::test_create_experiment` | **VERIFIED (PASSED)** |
| **FR-09** | §3.9 | [app/services/fl_service.py](file:///c:/Users/HP/Flex-ID/app/services/fl_service.py)<br>[app/api/v1/experiments.py](file:///c:/Users/HP/Flex-ID/app/api/v1/experiments.py) | `GET /api/v1/experiments`<br>`GET /api/v1/experiments/{id}/metrics` | `tests/test_experiments.py::test_list_experiments`<br>`tests/test_experiments.py::test_get_experiment_details_and_metrics` | **VERIFIED (PASSED)** |
| **FR-10** | §3.10| [app/services/explanation_service.py](file:///c:/Users/HP/Flex-ID/app/services/explanation_service.py)<br>[app/api/v1/explanations.py](file:///c:/Users/HP/Flex-ID/app/api/v1/explanations.py) | `POST /api/v1/explanations` | `tests/test_explanations.py::test_explain_prediction_endpoint` | **VERIFIED (PASSED)** |

---

## 2. Non-Functional Requirements Traceability

| NFR ID | Category | Implementation Mechanism | Test / Verification Method | Status |
| :---: | :--- | :--- | :--- | :---: |
| **NFR-01** | Performance | `ModelManager.warmup()`, in-memory Keras model cache | Measured latency $<3$ ms per inference in unit tests | **VERIFIED** |
| **NFR-02** | Validation | Pydantic v2 schemas (`TrafficFeaturesInput`, `BatchTrafficInput`) | `tests/test_predictions.py::test_prediction_invalid_feature_length` (422) | **VERIFIED** |
| **NFR-03** | Resilience | `np.nan_to_num`, bounds clipping to `[0.0, 1.0]` | `tests/test_inference_service.py::test_inference_service_nan_handling` | **VERIFIED** |
| **NFR-04** | Security | Centralized exception handlers, RCE elimination | Automated error sanitization tests, verified Docker user (UID 1001) | **VERIFIED** |
| **NFR-05** | Persistence | Dual SQLite / MySQL support via SQLAlchemy ORM | In-memory session tests across all endpoints | **VERIFIED** |
| **NFR-06** | Testing | Full pytest integration test suite | 19 passing tests in 2.53s | **VERIFIED** |
