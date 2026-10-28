from fastapi.testclient import TestClient

def test_list_models(client: TestClient):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["total_models"] >= 4
    assert data["active_model"] == "fedprox-v1"
    assert any(m["model_version"] == "fedprox-v1" for m in data["models"])
    assert any(m["model_version"] == "fedavg-v1" for m in data["models"])

def test_get_model_details(client: TestClient):
    response = client.get("/api/v1/models/fedprox-v1")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "fedprox-v1"
    assert data["strategy"] == "fedprox"
    assert data["accuracy"] == 0.9093
    assert data["is_active"] is True
    assert len(data["class_names"]) >= 2

def test_get_nonexistent_model(client: TestClient):
    response = client.get("/api/v1/models/nonexistent-model-xyz")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data or "message" in data

def test_activate_model(client: TestClient):
    response = client.post("/api/v1/models/fedavg-v1/activate")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "fedavg-v1"
    assert data["is_active"] is True

    # Restore fedprox-v1 as active
    resp2 = client.post("/api/v1/models/fedprox-v1/activate")
    assert resp2.status_code == 200
    assert resp2.json()["is_active"] is True
