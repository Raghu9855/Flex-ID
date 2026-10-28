from fastapi.testclient import TestClient

def test_root_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["database_connected"] is True
    assert "components" in data
    assert data["components"]["model_manager"] == "ready"

def test_api_v1_health_check(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"
    assert data["active_model"] == "fedprox-v1"
