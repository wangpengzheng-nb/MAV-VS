from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件 (server.py 从 web_app/ 启动时, 需指向上级目录)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autovs.pipeline import PipelineService
from autovs.library import SmiFormatError
from autovs.schemas import PocketSpec, TaskRequest
from autovs.security import ensure_within
from src.agents.target_scout import TargetScoutAgent
from src.agents.target_research.models import TargetIdentity, TargetIntent


STATIC_DIR = Path(__file__).resolve().parent / "static"
service = PipelineService()
app = FastAPI(title="AutoVS-Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def cache_policy(request: Request, call_next):
    """Prevent mixed frontend versions while retaining safe immutable asset caching."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    elif path.startswith("/static/"):
        if request.url.query:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def _asset_digest(filename: str) -> str:
    return hashlib.sha256((STATIC_DIR / filename).read_bytes()).hexdigest()[:12]


@app.get("/")
def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("__STYLE_VERSION__", _asset_digest("style.css"))
    html = html.replace("__APP_VERSION__", _asset_digest("app.js"))
    return HTMLResponse(html)


@app.get("/api/health")
def health():
    from autovs.capabilities import health_report
    return health_report(service.settings)


@app.get("/api/tasks")
def list_tasks(limit: int = 20):
    return {"tasks": service.store.list_tasks(limit)}


class TargetResolveRequest(BaseModel):
    query: str = Field(min_length=3, max_length=5000)
    target_hint: str = Field(default="", max_length=200)
    selected_accession: str = Field(default="", max_length=20)


@app.post("/api/targets/resolve")
def resolve_target(payload: TargetResolveRequest):
    result = TargetScoutAgent().resolve_target(
        payload.query, target_hint=payload.target_hint,
        selected_accession=payload.selected_accession,
    )
    if result.get("status") == "invalid_selection":
        raise HTTPException(422, result)
    return result


@app.post("/api/tasks")
async def create_task(
    request: Request,
    query: str = Form(...),
    target_gene: str = Form(""),
    target_uniprot_id: str = Form(""),
    center: str = Form(""), size: str = Form("24,24,24"), key_residues: str = Form(""),
    ligand_id: str = Form(""),
    ph: float = Form(7.4), cpu_only: bool = Form(False), baseline: bool = Form(False),
):
    if len(query.strip()) < 10:
        raise HTTPException(400, "任务描述至少10个字符")
    # 手动从 multipart form 中提取可选文件字段，
    # 避免 FastAPI File(None) 在某些版本/环境下将缺失字段误报为必填。
    raw_form = await request.form()
    protein = raw_form.get("protein")
    library = raw_form.get("library")
    if baseline and protein is None:
        raise HTTPException(422, "基础链路诊断模式必须上传预处理后的 PDB 文件")
    upload_dir = Path(tempfile.mkdtemp(prefix="autovs_upload_", dir=service.settings.task_root))
    protein_path = None
    library_path = None
    if protein is not None:
        suffix = Path(protein.filename or "").suffix.lower()
        if suffix != ".pdb":
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(422, "蛋白结构只接受 .pdb 文件")
        protein_path = upload_dir / "target_structure.pdb"
        with protein_path.open("wb") as out:
            shutil.copyfileobj(protein.file, out)
    if library is not None:
        suffix = Path(library.filename or "").suffix.lower()
        if suffix not in {".smi", ".smiles"}:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(422, "分子库只接受 .smi 或 .smiles 文件，格式为 molecule_id<TAB>SMILES")
        library_path = upload_dir / f"screening_library{suffix}"
        with library_path.open("wb") as out:
            shutil.copyfileobj(library.file, out)
    try:
        center_value = tuple(float(x) for x in center.split(",")) if center.strip() else None
        size_value = tuple(float(x) for x in size.split(","))
        if center_value is not None and len(center_value) != 3 or len(size_value) != 3:
            raise ValueError("center and size require three comma-separated numbers")
        verified_identity = None
        screening_intent = None
        if target_uniprot_id.strip() and not baseline:
            resolution = TargetScoutAgent().resolve_target(
                query, target_hint=target_gene.strip(), selected_accession=target_uniprot_id.strip(),
            )
            if resolution.get("status") != "resolved":
                raise ValueError(resolution.get("reason", "靶点身份验证失败"))
            verified_identity = TargetIdentity.model_validate(resolution["identity"])
            screening_intent = TargetIntent.model_validate(resolution["intent"])
        request = TaskRequest(query=query, protein_path=str(protein_path) if protein_path else None,
                              library_path=str(library_path) if library_path else None,
                              protein_original_name=protein.filename if protein else None,
                              library_original_name=library.filename if library else None,
                              pocket=PocketSpec(center=center_value, size=size_value,
                                                key_residues=[x.strip() for x in key_residues.split(",") if x.strip()],
                                                cocrystal_ligand=ligand_id.strip() or None),
                              ph=ph, cpu_only=cpu_only,
                              target_identity=verified_identity, screening_intent=screening_intent)
        task_id = service.submit(request, use_llm_planning=not baseline)
    except SmiFormatError as exc:
        raise HTTPException(422, exc.as_dict()) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)
    task = service.get_task(task_id) or {}
    manifest = task.get("input_manifest", {})
    return {
        "task_id": task_id, "status": "pending",
        "warnings": manifest.get("warnings", []),
        "input_summary": {
            "library_source": manifest.get("library_asset", {}).get("source"),
            "library_version": manifest.get("library_asset", {}).get("version"),
            "target_source": manifest.get("target_asset", {}).get("source"),
        },
    }


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return task


@app.get("/api/tasks/{task_id}/jobs/{job_id}/diagnostics")
def job_diagnostics(task_id: str, job_id: str):
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    job = next((item for item in task.get("jobs", []) if item["job_id"] == job_id), None)
    if not job:
        raise HTTPException(404, "工具任务不存在")
    artifacts = []
    snippets = []
    for artifact in task.get("artifacts", []):
        if artifact.get("job_id") != job_id:
            continue
        item = dict(artifact)
        item["download_url"] = f"/api/tasks/{task_id}/artifacts/{artifact['artifact_id']}"
        item.pop("path", None)
        artifacts.append(item)
        if artifact.get("format", "").upper() not in {"LOG", "TXT", "JSON"} or len(snippets) >= 4:
            continue
        path = ensure_within(artifact["path"], [service.settings.task_root], must_exist=True)
        text = path.read_text(encoding="utf-8", errors="replace")
        snippets.append({"name": artifact["name"], "content": text[-40_000:], "truncated": len(text) > 40_000})
    return {"job": job, "artifacts": artifacts, "snippets": snippets}


@app.delete("/api/tasks/{task_id}")
def cancel_task(task_id: str, permanent: bool = False):
    """取消正在运行或等待中的任务；permanent=true 时永久删除已完成/失败/已取消的任务。"""
    if permanent:
        try:
            result = service.delete_task(task_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return result
    ok = service.cancel_task(task_id)
    if not ok:
        raise HTTPException(409, "任务不存在或已结束，无法取消")
    return {"task_id": task_id, "status": "cancelled"}


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str):
    try:
        service.resume(task_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"task_id": task_id, "status": "running"}


@app.post("/api/tasks/{task_id}/pause")
def pause_task(task_id: str):
    """暂停正在运行的任务，保留已完成阶段的成果。"""
    ok = service.pause_task(task_id)
    if not ok:
        raise HTTPException(409, "任务不在运行状态，无法暂停")
    return {"task_id": task_id, "status": "paused"}


@app.get("/api/progress/{task_id}")
async def progress(task_id: str):
    async def events():
        while True:
            task = service.get_task(task_id)
            if not task:
                yield f"data: {json.dumps({'status':'error','message':'任务不存在'}, ensure_ascii=False)}\n\n"; return
            payload = _progress_payload(task)
            terminal = task["status"] in {"succeeded", "failed", "cancelled"}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if terminal:
                return
            await asyncio.sleep(2)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/result/{task_id}")
def result(task_id: str):
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return {"status": "done" if task["status"] == "succeeded" else task["status"], "result": task.get("result"), "error": task.get("error")}


@app.get("/api/tasks/{task_id}/artifacts/{artifact_id}")
def download_artifact(task_id: str, artifact_id: int):
    artifact = next((item for item in service.store.list_artifacts(task_id) if item["artifact_id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(404, "产物不存在")
    path = ensure_within(artifact["path"], [service.settings.task_root], must_exist=True)
    return FileResponse(path, filename=path.name)


def _progress_payload(task: dict) -> dict:
    phases = task.get("progress", [])
    counted = [phase for phase in phases if not (
        phase["status"] == "skipped"
        and (phase.get("message", "").startswith("基础链路") or phase.get("message", "").startswith("未包含")
             or phase.get("message", "").startswith("已锁定用户"))
    )]
    completed = sum(phase["status"] in {"succeeded", "failed", "quarantined", "cancelled"} for phase in counted)
    percent = int(100 * completed / max(1, len(counted)))
    if task["status"] == "succeeded":
        percent = 100
    current = next((phase for phase in phases if phase["status"] == "running"), None)
    if current is None:
        current = next((phase for phase in reversed(phases) if phase["status"] == "failed"), None)
    if current is None:
        current = next((phase for phase in reversed(phases) if phase["status"] == "succeeded"), None)
    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "percent": percent,
        "current_phase": current,
        "phases": phases,
        "jobs": task.get("jobs", []),
        "error": task.get("error", ""),
        "updated_at": task.get("updated_at"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
