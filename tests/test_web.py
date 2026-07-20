import re

from fastapi.testclient import TestClient

from web_app.server import _progress_payload, app


def test_health_endpoint_is_persistent_service():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert "capabilities" in response.json()


def test_console_and_persistent_task_list_are_available():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "执行时间线" in page.text
    assert "诊断面板" in page.text
    assert "__STYLE_VERSION__" not in page.text
    assert "__APP_VERSION__" not in page.text
    style_url = re.search(r'href="([^"]*style\.css\?v=[a-f0-9]{12})"', page.text).group(1)
    app_url = re.search(r'src="([^"]*app\.js\?v=[a-f0-9]{12})"', page.text).group(1)
    assert page.headers["cache-control"] == "no-store, max-age=0"
    assert "immutable" in client.get(style_url).headers["cache-control"]
    assert "immutable" in client.get(app_url).headers["cache-control"]
    assert "must-revalidate" in client.get("/static/style.css").headers["cache-control"]
    response = client.get("/api/tasks?limit=3")
    assert response.status_code == 200
    assert isinstance(response.json()["tasks"], list)
    assert response.headers["cache-control"] == "no-store, max-age=0"


def test_progress_payload_preserves_failed_stage_and_real_percent():
    task = {
        "task_id": "task-1", "status": "failed", "error": "tool failed", "updated_at": "now", "jobs": [],
        "progress": [
            {"phase_id": "input", "label": "输入校验", "status": "succeeded", "message": "done", "error": ""},
            {"phase_id": "docking", "label": "对接", "status": "failed", "message": "failed", "error": "smina unavailable"},
            {"phase_id": "report", "label": "报告", "status": "skipped", "message": "upstream failure", "error": ""},
        ],
    }
    payload = _progress_payload(task)
    assert payload["status"] == "failed"
    assert payload["percent"] == 66
    assert payload["current_phase"]["phase_id"] == "docking"
    assert payload["current_phase"]["error"] == "smina unavailable"
