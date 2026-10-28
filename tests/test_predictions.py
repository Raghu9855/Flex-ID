from fastapi.testclient import TestClient

def test_single_prediction_dict_input(client: TestClient, sample_benign_features: dict):
    payload = {
        "features": sample_benign_features,
        "model_version": "fedprox-v1",
    }
    response = client.post("/api/v1/predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["model_version"] == "fedprox-v1"
    assert "predicted_class" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert "is_intrusion" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0
    assert len(data["class_probabilities"]) > 0

    # Verify stored in database
    pred_id = data["prediction_id"]
    get_resp = client.get(f"/api/v1/predictions/{pred_id}")
    assert get_resp.status_code == 200
    log_data = get_resp.json()
    assert log_data["predicted_class"] == data["predicted_class"]

def test_single_prediction_list_input(client: TestClient, sample_feature_list: list):
    payload = {
        "features": sample_feature_list,
    }
    response = client.post("/api/v1/predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class_id"] >= 0

def test_prediction_invalid_feature_length(client: TestClient):
    payload = {
        "features": [0.1, 0.2, 0.3],  # Only 3 values instead of 29
    }
    response = client.post("/api/v1/predictions", json=payload)
    assert response.status_code == 422  # Pydantic validation error

def test_batch_prediction(client: TestClient, sample_feature_list: list):
    payload = {
        "records": [
            {"features": sample_feature_list},
            {"features": [0.1] * 29},
            {"features": [0.2] * 29},
        ],
        "model_version": "fedprox-v1",
    }
    response = client.post("/api/v1/predictions/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["predictions"]) == 3
    assert "total_latency_ms" in data
    assert isinstance(data["intrusions_detected"], int)
