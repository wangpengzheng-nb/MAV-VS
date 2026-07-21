from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from autovs.capabilities import health_report, list_capabilities
from autovs.compiler import compile_strategy, validate_workflow_bindings
from autovs.config import load_settings
from autovs.db import StateStore
from autovs.schemas import InputManifest, JobStatus, WorkflowPlan, WorkflowStep
from autovs.tool_manager import ToolManager


def create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError('MCP SDK is not installed; install with: pip install "mcp[cli]>=1.27,<2"') from exc

    settings = load_settings()
    store = StateStore(settings.database_path)
    manager = ToolManager(settings, store)
    mcp = FastMCP("autovs_tools_mcp", host=settings.host, port=settings.port, json_response=True)

    @mcp.tool(name="autovs_list_capabilities", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    def autovs_list_capabilities() -> list[dict[str, Any]]:
        """List the finite, registered AutoVS computation capabilities and current availability."""
        return [item.model_dump(mode="json") for item in list_capabilities(settings)]

    @mcp.tool(name="autovs_health_check", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    def autovs_health_check() -> dict[str, Any]:
        """Check configured executables, environments, database, containers, and capability degradation."""
        return health_report(settings)

    @mcp.tool(name="autovs_validate_workflow", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    def autovs_validate_workflow(workflow: dict[str, Any], input_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate WorkflowPlan v1 plus optional immutable input bindings, or compile one evolved strategy."""
        try:
            manifest = InputManifest.model_validate(input_manifest) if input_manifest is not None else None
            if "plan_version" in workflow:
                plan = WorkflowPlan.model_validate(workflow)
                if manifest is not None:
                    validate_workflow_bindings(plan, manifest)
            else:
                plan = compile_strategy(workflow, input_manifest=manifest)
            return {"valid": True, "workflow": plan.model_dump(mode="json")}
        except Exception as exc:
            return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool(name="autovs_submit_step", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True})
    def autovs_submit_step(task_id: str, step: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        """Submit one registered step; network access, when needed, is limited to the controlled RCSB adapter."""
        job = manager.submit(task_id, WorkflowStep.model_validate(step), inputs, background=True)
        return job.model_dump(mode="json")

    @mcp.tool(name="autovs_get_job", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    def autovs_get_job(job_id: str) -> dict[str, Any]:
        """Get persisted local/Slurm job status and the latest actionable message."""
        job = store.get_job(job_id)
        return job.model_dump(mode="json") if job else {"error": "job not found"}

    @mcp.tool(name="autovs_list_artifacts", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    def autovs_list_artifacts(task_id: str) -> list[dict[str, Any]]:
        """List checksummed artifacts produced for one task."""
        return store.list_artifacts(task_id)

    @mcp.tool(name="autovs_get_job_log", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    def autovs_get_job_log(job_id: str, tail_chars: int = 8000) -> dict[str, Any]:
        """Return the persisted job message; filesystem paths are not accepted from callers."""
        job = store.get_job(job_id)
        return {"job_id": job_id, "status": job.status.value, "log": job.message[-max(1, min(tail_chars, 50000)):]} if job else {"error": "job not found"}

    @mcp.tool(name="autovs_cancel_job", annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False})
    def autovs_cancel_job(job_id: str, confirmed: bool = False) -> dict[str, Any]:
        """Cancel a job only after explicit user confirmation; cancels Slurm job when present."""
        if not confirmed:
            return {"cancelled": False, "error": "explicit confirmed=true is required"}
        job = store.get_job(job_id)
        if not job:
            return {"cancelled": False, "error": "job not found"}
        if job.slurm_job_id:
            scancel = settings.executable("scancel")
            if not scancel or not scancel.exists():
                return {"cancelled": False, "error": "scancel is unavailable"}
            subprocess.run([str(scancel), job.slurm_job_id], check=False, shell=False)
        store.update_job(job_id, JobStatus.CANCELLED, message="cancelled by explicit user request")
        return {"cancelled": True, "job_id": job_id}

    return mcp


def main() -> None:
    create_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
