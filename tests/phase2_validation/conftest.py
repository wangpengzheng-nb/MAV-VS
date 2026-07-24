"""Phase 2 验证测试共享 fixture."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_existing_baseline_task() -> Path | None:
    """查找最近成功的baseline任务目录，复用其产物."""
    tasks_root = PROJECT_ROOT / "runtime" / "tasks"
    if not tasks_root.is_dir():
        return None
    for task_dir in sorted(tasks_root.iterdir(), reverse=True):
        if not task_dir.is_dir():
            continue
        plan = task_dir / "workflow_plan.json"
        state = task_dir / "workflow_execution_state.json"
        if plan.is_file() and state.is_file():
            try:
                es = json.loads(state.read_text())
                if es.get("artifact_state", {}).get("docked_poses"):
                    return task_dir
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def get_baseline_artifacts(task_dir: Path | None = None) -> dict[str, Path]:
    """提取baseline任务的产物路径."""
    if task_dir is None:
        task_dir = get_existing_baseline_task()
    if task_dir is None:
        return {}

    steps = task_dir / "steps"
    artifacts: dict[str, Path] = {}

    # 受体文件
    for p in [
        steps / "protein-preparation" / "receptor.pdbqt",
        steps / "protein-preparation" / "receptor_clean.pdb",
    ]:
        if p.is_file():
            artifacts[p.name] = p

    # 配体文件
    for p in [
        steps / "molecule-standardization" / "prepared_library.sdf",
        steps / "format-conversion" / "converted.sdf",
        steps / "physicochemical-filtering" / "prepared_library.sdf",
    ]:
        if p.is_file():
            artifacts["prepared_library.sdf"] = p
            break

    # Manifest
    for p in [
        steps / "molecule-standardization" / "manifest.csv",
        steps / "physicochemical-filtering" / "manifest.csv",
    ]:
        if p.is_file():
            artifacts["manifest.csv"] = p
            break

    # 对接结果
    for p in [
        steps / "molecular-docking" / "smina_poses.sdf",
        steps / "molecular-docking" / "gnina_poses.sdf",
    ]:
        if p.is_file():
            artifacts[p.name] = p

    # 打分CSV
    for p in [
        steps / "molecular-docking" / "smina_scores.csv",
        steps / "molecular-docking" / "gnina_scores.csv",
        steps / "molecular-docking" / "combined_scores.csv",
        task_dir / "combined_scores.csv",
    ]:
        if p.is_file():
            artifacts["scores.csv"] = p
            break

    # 姿态/复合物
    for p in [
        steps / "pose-extraction" / "selected_poses.sdf",
        steps / "pose-extraction" / "complex_index.csv",
    ]:
        if p.is_file():
            artifacts[p.name] = p

    # 口袋
    pocket_path = steps / "pocket-definition" / "pocket.json"
    if pocket_path.is_file():
        artifacts["pocket.json"] = pocket_path

    return artifacts


def get_pocket_params(task_dir: Path | None = None) -> dict[str, Any]:
    """从已有任务获取口袋参数."""
    artifacts = get_baseline_artifacts(task_dir)
    pocket_json = artifacts.get("pocket.json")
    if pocket_json:
        try:
            data = json.loads(pocket_json.read_text())
            pocket = data.get("selected_pocket", {})
            return {
                "center": tuple(pocket.get("center", (-15.36, 2.24, -9.56))),
                "size": tuple(pocket.get("size", (24, 24, 24))),
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return {"center": (-15.36, 2.24, -9.56), "size": (24, 24, 24)}


def submit_slurm_job(script_path: Path) -> str:
    """提交 Slurm 作业，返回 job_id."""
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True, timeout=30, cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split()
        return parts[-1] if parts else ""
    raise RuntimeError(f"sbatch failed: {result.stderr}")


def poll_slurm_job(job_id: str, timeout: int = 36000, interval: int = 60) -> dict[str, Any]:
    """轮询 Slurm 作业直到完成或超时."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-o", "%T", "-h", "--noheader"],
            capture_output=True, text=True, timeout=30,
        )
        state = result.stdout.strip()
        if state in ("COMPLETED",):
            return {"status": "completed", "job_id": job_id}
        if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
            return {"status": "failed", "job_id": job_id, "state": state}
        if not state:
            # 作业已不在队列，假设完成
            return {"status": "completed", "job_id": job_id}
        time.sleep(interval)
    return {"status": "timeout", "job_id": job_id}


def cancel_slurm_job(job_id: str) -> bool:
    """取消 Slurm 作业."""
    result = subprocess.run(
        ["scancel", job_id], capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0
