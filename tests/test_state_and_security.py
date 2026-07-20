from pathlib import Path

import pytest

from autovs.db import StateStore
from autovs.schemas import JobStatus
from autovs.security import SecurityError, ensure_within


def test_state_survives_new_store_instance(tmp_path):
    path = tmp_path / "state.sqlite3"; task_dir = tmp_path / "task"; task_dir.mkdir()
    first = StateStore(path); task_id = first.create_task({"query": "a sufficiently long query"}, task_dir)
    first.update_task(task_id, JobStatus.RUNNING)
    assert StateStore(path).get_task(task_id)["status"] == "running"


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
