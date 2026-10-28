from fastapi.testclient import TestClient

def test_explain_prediction_endpoint(client: TestClient, sample_feature_list: list):
    payload = {
        "features": sample_feature_list,
        "model_version": "fedprox-v1",
        "top_k": 5,
    }
    response = client.post("/api/v1/explanations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) == 5
    for item in data["top_contributing_features"]:
        assert "feature" in item
        assert "importance" in item
        assert item["direction"] in ("positive", "negative")
    assert "summary" in data
    assert len(data["summary"]) > 10
