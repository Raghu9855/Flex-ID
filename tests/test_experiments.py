from fastapi.testclient import TestClient

def test_list_experiments(client: TestClient):
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["total_experiments"] >= 4
    assert len(data["experiments"]) >= 4

def test_get_experiment_details_and_metrics(client: TestClient):
    # First get list to find an existing experiment ID
    list_resp = client.get("/api/v1/experiments")
    experiments = list_resp.json()["experiments"]
    exp_id = experiments[0]["id"]

    # Get details
    resp = client.get(f"/api/v1/experiments/{exp_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == exp_id
    assert "strategy" in detail
    assert "status" in detail

    # Get metrics
    metrics_resp = client.get(f"/api/v1/experiments/{exp_id}/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert isinstance(metrics, list)

def test_create_experiment(client: TestClient):
    payload = {
        "name": "Integration Test Experiment",
        "strategy": "fedprox",
        "rounds": 3,
        "num_clients": 4,
        "proximal_mu": 0.1,
        "is_attack": False,
    }
    response = client.post("/api/v1/experiments", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["name"] == "Integration Test Experiment"
    assert data["strategy"] == "fedprox"
    assert data["status"] in ("running", "completed")

def test_get_nonexistent_experiment(client: TestClient):
    response = client.get("/api/v1/experiments/invalid-uuid-123")
    assert response.status_code == 404
