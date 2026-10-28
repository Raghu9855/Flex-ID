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

def test_serve_ui_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "FLEX-ID" in response.text

def test_serve_ui_static_assets(client: TestClient):
    css_res = client.get("/css/style.css")
    assert css_res.status_code == 200
    assert "var(--accent)" in css_res.text

    js_res = client.get("/js/app.js")
    assert js_res.status_code == 200
    assert "CANONICAL_FEATURES" in js_res.text
