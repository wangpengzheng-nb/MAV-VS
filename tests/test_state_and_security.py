from pathlib import Path

import pytest

from autovs.db import StateStore
from autovs.config import Settings
from autovs.schemas import ActionType, JobStatus, WorkflowStep
from autovs.security import SecurityError, ensure_within
from autovs.tool_manager import ToolManager


def test_state_survives_new_store_instance(tmp_path):
    path = tmp_path / "state.sqlite3"; task_dir = tmp_path / "task"; task_dir.mkdir()
    first = StateStore(path); task_id = first.create_task({"query": "a sufficiently long query"}, task_dir)
    first.update_task(task_id, JobStatus.RUNNING)
    assert StateStore(path).get_task(task_id)["status"] == "running"


def test_persisted_request_can_be_migrated_without_losing_task_identity(tmp_path):
    path = tmp_path / "state.sqlite3"; task_dir = tmp_path / "task"; task_dir.mkdir()
    store = StateStore(path); task_id = store.create_task({"query": "a sufficiently long query"}, task_dir)
    store.update_task_request(task_id, {"query": "a sufficiently long query", "input_manifest_path": "manifest.json"})
    task = StateStore(path).get_task(task_id)
    assert task["task_id"] == task_id
    assert task["request"]["input_manifest_path"] == "manifest.json"


def test_progress_events_are_persistent_and_diagnostic(tmp_path):
    path = tmp_path / "state.sqlite3"; task_dir = tmp_path / "task"; task_dir.mkdir()
    store = StateStore(path); task_id = store.create_task({"query": "a sufficiently long query"}, task_dir)
    store.initialize_progress(task_id, [("research", "靶点调研"), ("docking", "分子对接")])
    store.update_progress(task_id, "research", JobStatus.SUCCEEDED, message="done")
    store.update_progress(task_id, "docking", JobStatus.RUNNING, message="running", metadata={"job_id": "job-1"})
    store.fail_running_progress(task_id, error="smina unavailable")
    progress = StateStore(path).list_progress(task_id)
    assert [item["status"] for item in progress] == ["succeeded", "failed"]
    assert progress[1]["metadata"]["job_id"] == "job-1"
    assert progress[1]["error"] == "smina unavailable"


def test_path_allowlist_blocks_escape(tmp_path):
    root = tmp_path / "root"; root.mkdir(); inside = root / "a.txt"; inside.write_text("x")
    assert ensure_within(inside, [root], must_exist=True) == inside.resolve()
    with pytest.raises(SecurityError):
        ensure_within(tmp_path / "outside.txt", [root])


def test_structure_acquisition_rejects_caller_supplied_url(tmp_path):
    task_root = tmp_path / "tasks"; task_dir = task_root / "task"; task_dir.mkdir(parents=True)
    settings = Settings(raw={
        "service": {"database": str(tmp_path / "state.sqlite3"), "task_root": str(task_root),
                    "host": "127.0.0.1", "port": 8765},
        "executables": {}, "limits": {}, "environments": {}, "containers": {},
    }, config_path=tmp_path / "tools.toml")
    store = StateStore(settings.database_path); task_id = store.create_task({"query": "long enough query"}, task_dir)
    research = task_dir / "research.json"; research.write_text("{}")
    manager = ToolManager(settings, store)
    job = manager.submit(task_id, WorkflowStep(
        step_id="acquire", action_type=ActionType.TARGET_STRUCTURE_ACQUISITION,
    ), {"research_path": str(research), "url": "https://evil.invalid/payload"}, background=False)
    assert job.status == JobStatus.FAILED
    assert "unsupported inputs" in job.message
