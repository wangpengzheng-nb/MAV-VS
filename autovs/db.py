from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from autovs.schemas import ActionType, JobRecord, JobStatus


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, status TEXT NOT NULL, request_json TEXT NOT NULL,
                    task_dir TEXT NOT NULL, result_json TEXT, error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, step_id TEXT NOT NULL,
                    action_type TEXT NOT NULL, status TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
                    slurm_job_id TEXT, message TEXT NOT NULL DEFAULT '', command_json TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    job_id TEXT, name TEXT NOT NULL, path TEXT NOT NULL, format TEXT NOT NULL,
                    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_task ON jobs(task_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
                """
            )

    def create_task(self, request: dict[str, Any], task_dir: Path) -> str:
        task_id = uuid.uuid4().hex[:16]
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, NULL, '', ?, ?)",
                (task_id, JobStatus.PENDING.value, json.dumps(request, ensure_ascii=False), str(task_dir), now, now),
            )
        return task_id

    def update_task(self, task_id: str, status: JobStatus, *, result: dict | None = None, error: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, result_json=COALESCE(?, result_json), error=?, updated_at=? WHERE task_id=?",
                (status.value, json.dumps(result, ensure_ascii=False) if result is not None else None, error, utcnow(), task_id),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["request"] = json.loads(data.pop("request_json"))
        data["result"] = json.loads(data.pop("result_json")) if data.get("result_json") else None
        return data

    def create_job(self, task_id: str, step_id: str, action_type: ActionType, command: Any = None) -> JobRecord:
        job_id, now = uuid.uuid4().hex[:16], utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, 0, NULL, '', ?, ?, ?)",
                (job_id, task_id, step_id, action_type.value, JobStatus.PENDING.value,
                 json.dumps(command or {}, sort_keys=True), now, now),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(self, job_id: str, status: JobStatus, *, message: str = "", slurm_job_id: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, message=?, slurm_job_id=COALESCE(?, slurm_job_id), updated_at=? WHERE job_id=?",
                (status.value, message, slurm_job_id, utcnow(), job_id),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data.pop("command_json", None)
        return JobRecord.model_validate(data)

    def find_checkpoint_job(self, task_id: str, step_id: str, checkpoint_key: str) -> JobRecord | None:
        encoded = json.dumps({"checkpoint_key": checkpoint_key}, sort_keys=True)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE task_id=? AND step_id=? AND status=? AND command_json=? ORDER BY updated_at DESC LIMIT 1",
                (task_id, step_id, JobStatus.SUCCEEDED.value, encoded),
            ).fetchone()
        return self.get_job(row["job_id"]) if row else None

    def list_jobs(self, task_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM jobs WHERE task_id=? ORDER BY created_at", (task_id,))]

    def add_artifact(self, task_id: str, job_id: str | None, name: str, path: Path, fmt: str, sha256: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO artifacts(task_id,job_id,name,path,format,sha256,size_bytes,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (task_id, job_id, name, str(path), fmt, sha256, path.stat().st_size, utcnow()),
            )

    def list_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM artifacts WHERE task_id=? ORDER BY artifact_id", (task_id,))]
