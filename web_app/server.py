from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from autovs.pipeline import PipelineService
from autovs.schemas import PocketSpec, TaskRequest
from autovs.security import ensure_within


STATIC_DIR = Path(__file__).resolve().parent / "static"
service = PipelineService()
app = FastAPI(title="AutoVS-Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    from autovs.capabilities import health_report
    return health_report(service.settings)


@app.post("/api/tasks")
async def create_task(
    query: str = Form(...), protein: UploadFile = File(...), library: UploadFile = File(...),
    center: str = Form(""), size: str = Form("24,24,24"), key_residues: str = Form(""),
    ph: float = Form(7.4), cpu_only: bool = Form(False), baseline: bool = Form(False),
):
    if len(query.strip()) < 10:
        raise HTTPException(400, "任务描述至少10个字符")
    upload_dir = Path(tempfile.mkdtemp(prefix="autovs_upload_", dir=service.settings.task_root))
    protein_path = upload_dir / f"protein{Path(protein.filename or '.pdb').suffix.lower()}"
    library_path = upload_dir / f"library{Path(library.filename or '.smi').suffix.lower()}"
    with protein_path.open("wb") as out:
        shutil.copyfileobj(protein.file, out)
    with library_path.open("wb") as out:
        shutil.copyfileobj(library.file, out)
    try:
        center_value = tuple(float(x) for x in center.split(",")) if center.strip() else None
        size_value = tuple(float(x) for x in size.split(","))
        if center_value is not None and len(center_value) != 3 or len(size_value) != 3:
            raise ValueError("center and size require three comma-separated numbers")
        request = TaskRequest(query=query, protein_path=str(protein_path), library_path=str(library_path),
                              pocket=PocketSpec(center=center_value, size=size_value,
                                                key_residues=[x.strip() for x in key_residues.split(",") if x.strip()]),
                              ph=ph, cpu_only=cpu_only)
        task_id = service.submit(request, use_llm_planning=not baseline)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return task


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str):
    try:
        service.resume(task_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"task_id": task_id, "status": "running"}


@app.get("/api/progress/{task_id}")
async def progress(task_id: str):
    async def events():
        while True:
            task = service.get_task(task_id)
            if not task:
                yield f"data: {json.dumps({'status':'error','message':'任务不存在'}, ensure_ascii=False)}\n\n"; return
            jobs = task.get("jobs", [])
            completed = sum(job["status"] in {"succeeded", "failed", "skipped", "quarantined", "cancelled"} for job in jobs)
            percent = min(99, int(100 * completed / max(1, len(jobs) + (0 if task["status"] in {"succeeded", "failed"} else 1))))
            terminal = task["status"] in {"succeeded", "failed"}
            payload = {"task_id": task_id, "status": "done" if task["status"] == "succeeded" else ("error" if task["status"] == "failed" else "running"),
                       "percent": 100 if terminal else percent, "step": jobs[-1]["step_id"] if jobs else "planning",
                       "message": task.get("error") or task["status"]}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
