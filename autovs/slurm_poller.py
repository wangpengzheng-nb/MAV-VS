"""Slurm 作业自动轮询器.

后台线程定期检测所有 PAUSED 任务中已提交的 Slurm 作业状态，
作业完成后自动恢复 pipeline 继续执行。实现真正的无人值守。

核心逻辑:
1. 每30秒扫描所有状态为 PAUSED 的任务
2. 对每个任务，查找其最新的 JobRecord（含 slurm_job_id）
3. 通过 squeue/sacct 查询 Slurm 作业状态
4. 如果完成(COMPLETED) → auto-resume
5. 如果失败(FAILED/TIMEOUT等) → 标记任务失败
6. 如果仍在运行 → 跳过，下轮再检查
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autovs.pipeline import PipelineService

logger = logging.getLogger(__name__)


class SlurmPoller:
    """Slurm 作业自动轮询器。"""

    def __init__(
        self,
        pipeline: PipelineService,
        interval: int = 30,
        max_retries: int = 100,
    ):
        self.pipeline = pipeline
        self.interval = interval
        self.max_retries = max_retries
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._retry_counts: dict[str, int] = {}

    def start(self) -> None:
        """启动后台轮询线程."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="slurm-poller")
        self._thread.start()
        logger.info("SlurmPoller started (interval=%ds)", self.interval)

    def stop(self) -> None:
        """停止后台轮询线程."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("SlurmPoller stopped")

    def _poll_loop(self) -> None:
        """主轮询循环."""
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("SlurmPoller error in poll cycle")
            self._stop.wait(self.interval)

    def _poll_once(self) -> None:
        """执行一次轮询扫描."""
        # 从数据库获取所有 PAUSED 任务
        paused_tasks = self._get_paused_tasks()
        if not paused_tasks:
            return

        for task in paused_tasks:
            task_id = task.get("task_id", "")
            if not task_id:
                continue

            # 获取任务最新的 PAUSED 作业
            paused_job = self._get_paused_job(task_id)
            if not paused_job:
                continue

            slurm_id = paused_job.get("slurm_job_id", "")
            if not slurm_id:
                # 检查 pending.json 是否有 slurm_job_id
                slurm_id = self._extract_slurm_id_from_state(task, paused_job)

            if not slurm_id:
                continue

            # 查询 Slurm 作业状态
            status = self._check_slurm_status(slurm_id)

            if status == "COMPLETED":
                logger.info("Slurm job %s completed, auto-resuming task %s", slurm_id, task_id)
                try:
                    self.pipeline.resume(task_id)
                    self._retry_counts.pop(task_id, None)
                except Exception:
                    logger.exception("Failed to auto-resume task %s", task_id)

            elif status in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
                logger.warning(
                    "Slurm job %s failed (%s), marking task %s as failed",
                    slurm_id, status, task_id,
                )
                try:
                    # 取消任务，清理资源
                    self.pipeline.cancel_task(task_id)
                    self._retry_counts.pop(task_id, None)
                except Exception:
                    logger.exception("Failed to cancel task %s", task_id)

            elif status == "PENDING":
                # 检查是否超时
                retries = self._retry_counts.get(task_id, 0) + 1
                self._retry_counts[task_id] = retries
                if retries > self.max_retries:
                    logger.warning(
                        "Task %s Slurm job %s pending too long (%d polls), cancelling",
                        task_id, slurm_id, retries,
                    )
                    self.pipeline.cancel_task(task_id)
                    self._retry_counts.pop(task_id, None)

            # RUNNING 或其他状态: 继续等待

    def _get_paused_tasks(self) -> list[dict]:
        """获取所有 PAUSED 状态的任务."""
        try:
            return self.pipeline.store.list_paused_tasks()
        except AttributeError:
            # 如果 store 没有此方法，手动筛选
            all_tasks = self.pipeline.store.list_tasks()
            return [t for t in all_tasks if t.get("status") == "paused"]

    def _get_paused_job(self, task_id: str) -> dict | None:
        """获取任务最新的 PAUSED 作业."""
        jobs = self.pipeline.store.list_jobs(task_id)
        paused = [j for j in jobs if j.get("status") == "paused"]
        if not paused:
            return None
        # 返回最新的
        paused.sort(key=lambda j: j.get("updated_at", ""), reverse=True)
        return paused[0]

    def _extract_slurm_id_from_state(self, task: dict, job: dict) -> str:
        """从 pending state 文件中提取 slurm_job_id."""
        task_dir = task.get("task_dir", "")
        if not task_dir:
            return ""
        step_id = job.get("step_id", "")
        pending_path = Path(task_dir) / "steps" / step_id / "pending.json"
        if pending_path.is_file():
            try:
                data = json.loads(pending_path.read_text(encoding="utf-8"))
                return str(data.get("slurm_job_id", ""))
            except (json.JSONDecodeError, OSError):
                pass
        return ""

    @staticmethod
    def _check_slurm_status(slurm_id: str) -> str:
        """查询 Slurm 作业状态."""
        # 先尝试 squeue
        try:
            result = subprocess.run(
                ["squeue", "-j", slurm_id, "-o", "%T", "-h", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            state = result.stdout.strip()
            if state:
                return state
        except (subprocess.TimeoutExpired, OSError):
            pass

        # squeue 未找到(已完成或不在队列)，用 sacct 查询
        try:
            result = subprocess.run(
                ["sacct", "-j", slurm_id, "-o", "State", "-n", "-P",
                 "--noheader", "-X"],
                capture_output=True, text=True, timeout=15,
            )
            states = result.stdout.strip().split()
            if states:
                # 取最后一个状态
                last = states[-1]
                status_map = {
                    "COMPLETED": "COMPLETED",
                    "FAILED": "FAILED",
                    "TIMEOUT": "TIMEOUT",
                    "CANCELLED": "CANCELLED",
                    "NODE_FAIL": "NODE_FAIL",
                    "OUT_OF_MEMORY": "OUT_OF_MEMORY",
                    "PENDING": "PENDING",
                    "RUNNING": "RUNNING",
                }
                return status_map.get(last, last)
        except (subprocess.TimeoutExpired, OSError):
            pass

        return "UNKNOWN"
