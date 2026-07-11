"""
Web Server — FastAPI + SSE 实时进度推送
=========================================
"""
from __future__ import annotations
import os, sys, json, uuid, queue, threading, hashlib
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from web_app.pipeline_runner import PipelineRunner

app = FastAPI(title="AutoVS-Agent Web", version="3.0")

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# 静态文件
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 任务输出目录
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "分析文件")

# 全局状态
progress_queues: Dict[str, queue.Queue] = {}
task_results: Dict[str, dict] = {}
task_status: Dict[str, str] = {}  # "running" | "done" | "error"


def make_task_dir(query: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(query.encode()).hexdigest()[:8]
    d = os.path.join(OUTPUT_BASE, f"任务_{ts}_{h}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "query.txt"), "w") as f:
        f.write(query)
    return d


# =========================================================================
# API 路由
# =========================================================================

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/run")
async def run_pipeline(request: Request):
    """启动流水线, 返回 task_id。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "任务描述不能为空")
    if len(query) < 10:
        raise HTTPException(400, "任务描述至少10个字符")

    task_id = str(uuid.uuid4())[:12]
    q = queue.Queue()
    progress_queues[task_id] = q
    task_status[task_id] = "running"

    task_dir = make_task_dir(query)

    def on_progress(step: str, status: str, percent: int, msg: str = ""):
        q.put({
            "task_id": task_id,
            "step": step,
            "status": status,
            "percent": percent,
            "message": msg,
        })

    def run_in_thread():
        try:
            runner = PipelineRunner(progress_callback=on_progress)
            result = runner.run(query, task_dir)
            task_results[task_id] = result
            task_status[task_id] = "done"
            # 发送完成信号
            q.put({
                "task_id": task_id,
                "step": "输出结果",
                "status": "done",
                "percent": 100,
                "message": "完成!",
            })
        except Exception as e:
            task_status[task_id] = "error"
            q.put({
                "task_id": task_id,
                "step": "错误",
                "status": "error",
                "percent": 0,
                "message": str(e)[:500],
            })
        finally:
            # 清理 queue 引用 (但保留10分钟供客户端读取)
            # queue 在 stream 结束后由垃圾回收处理
            pass

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started"}


@app.get("/api/progress/{task_id}")
async def stream_progress(task_id: str):
    """SSE 流式推送进度。"""
    async def generate():
        q = progress_queues.get(task_id)
        if q is None:
            status = task_status.get(task_id, "unknown")
            yield f"data: {json.dumps({'status': status, 'percent': 100 if status == 'done' else 0, 'step': '?', 'message': f'任务{status}'})}\n\n"
            return

        last_data = None
        while True:
            try:
                data = q.get(timeout=2)
                # 去重: 相同进度不重复推送
                key = (data.get("step"), data.get("percent"))
                if key == last_data:
                    continue
                last_data = key
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in ("done", "error") and data.get("percent", 0) >= 100:
                    break
            except queue.Empty:
                yield ": heartbeat\n\n"
                # 检查任务是否已完成但队列为空
                if task_status.get(task_id) in ("done", "error"):
                    break

        # 清理
        progress_queues.pop(task_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取最终结果。"""
    status = task_status.get(task_id, "unknown")
    if status == "running":
        return {"status": "running", "message": "任务仍在运行中"}
    if status == "error":
        return {"status": "error", "message": "任务执行失败"}
    result = task_results.get(task_id)
    if not result:
        raise HTTPException(404, "任务不存在")
    return {"status": "done", "result": result}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_tasks": sum(1 for s in task_status.values() if s == "running"),
        "completed_tasks": sum(1 for s in task_status.values() if s == "done"),
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AutoVS-Agent Web Server v3.0")
    print("  访问: http://localhost:8080")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
