from fastapi.testclient import TestClient

from web_app.server import app


def test_health_endpoint_is_persistent_service():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert "capabilities" in response.json()
