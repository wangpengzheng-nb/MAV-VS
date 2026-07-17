"""数据总线 — 文件生命周期管理、格式转换、资源监控、Checkpoint。"""
from __future__ import annotations
import os, json, hashlib, shutil, tempfile, time, threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# ═══════════════════════════════════════════
# ResourceContext — 全局资源跟踪
# ═══════════════════════════════════════════

class ResourceContext:
    """跟踪用户上传文件和中间产物, 工作流结束统一清理。"""

    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._files: Dict[str, Path] = {}
        self._temp_files: List[Path] = []

    def add_file(self, name: str, path: str):
        self._files[name] = Path(path)

    def get_file(self, name: str) -> Optional[Path]:
        return self._files.get(name)

    def register_temp(self, path: Path):
        self._temp_files.append(path)

    def new_temp(self, prefix: str = "tmp", suffix: str = "") -> Path:
        p = self.work_dir / f"{prefix}_{_short_ts()}{suffix}"
        self.register_temp(p)
        return p

    def cleanup(self):
        for p in self._temp_files:
            if p.exists():
                p.unlink(missing_ok=True)


# ═══════════════════════════════════════════
# DataBus — 格式转换 + Checkpoint
# ═══════════════════════════════════════════

FORMAT_CONVERTERS = {
    ("SDF", "PDBQT"):   "obabel {inp} -O {out} --gen3d",
    ("SMILES", "SDF"):  "obabel {inp} -O {out} --gen3d",
    ("PDB", "PDBQT"):   "obabel {inp} -O {out} -xr",
    ("SDF", "SMILES"):  "obabel {inp} -O {out}",
}


class DataBus:
    """连接各工具的数据通道: 格式转换 + Checkpoint 读写。"""

    def __init__(self, ctx: ResourceContext):
        self.ctx = ctx

    def convert(self, input_path: Path, from_fmt: str, to_fmt: str) -> Path:
        """格式转换, 自动插入转换节点。"""
        inp = str(input_path)
        out_path = self.ctx.new_temp(suffix=f".{to_fmt.lower()}")
        key = (from_fmt.upper(), to_fmt.upper())
        if key in FORMAT_CONVERTERS:
            cmd = FORMAT_CONVERTERS[key].format(inp=inp, out=str(out_path))
            _run_cmd(cmd)
            return out_path
        raise ValueError(f"不支持的格式转换: {from_fmt} → {to_fmt}")

    # ── Checkpoint ──

    def checkpoint_save(self, step_id: str, params: dict,
                         outputs: Dict[str, str],
                         resource_snapshot: Optional[dict] = None):
        ckpt = {
            "step_id": step_id,
            "status": "completed",
            "params_hash": _hash_dict(params),
            "params": params,
            "output_files": [{"key": k, "path": v,
                              "hash": _hash_file(v) if os.path.exists(v) else "?"}
                             for k, v in outputs.items()],
            "timestamp": datetime.now().isoformat(),
            "resource_snapshot": resource_snapshot or {},
        }
        ckpt_path = self.ctx.work_dir / f".checkpoint_{step_id}.json"
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)

    def checkpoint_load(self, step_id: str) -> Optional[dict]:
        ckpt_path = self.ctx.work_dir / f".checkpoint_{step_id}.json"
        if not ckpt_path.exists():
            return None
        with open(ckpt_path) as f:
            return json.load(f)

    def should_skip(self, step_id: str, current_params: dict) -> bool:
        """检查是否需要跳过: 参数未变+输出文件都存在。"""
        ckpt = self.checkpoint_load(step_id)
        if not ckpt or ckpt.get("status") != "completed":
            return False
        params_match = ckpt.get("params_hash") == _hash_dict(current_params)
        files_ok = all(os.path.exists(f["path"]) for f in ckpt.get("output_files", []))
        return params_match and files_ok


# ═══════════════════════════════════════════
# ResourceMonitor — 后台资源采样
# ═══════════════════════════════════════════

class ResourceMonitor(threading.Thread):
    """后台线程, 每30s采样资源并告警。"""

    def __init__(self, work_dir: str, sample_interval: float = 30.0):
        super().__init__(daemon=True)
        self.work_dir = work_dir
        self.interval = sample_interval
        self.running = True
        self.warnings: List[str] = []
        self.snapshots: List[dict] = []

    def run(self):
        while self.running:
            try:
                snap = self._sample()
                self.snapshots.append(snap)
                self._check(snap)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def last_snapshot(self) -> dict:
        return self.snapshots[-1] if self.snapshots else {}

    def _sample(self) -> dict:
        snap = {"time": datetime.now().isoformat()}
        try:
            import psutil
            snap["cpu_pct"] = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            snap["mem_used_gb"] = round(mem.used / 2**30, 1)
            snap["mem_avail_gb"] = round(mem.available / 2**30, 1)
            disk = psutil.disk_usage(self.work_dir)
            snap["disk_free_gb"] = round(disk.free / 2**30, 1)
        except ImportError:
            snap["note"] = "psutil未安装"
        try:
            import subprocess as sp
            r = sp.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                snap["gpu_pct"] = int(parts[0].strip()) if parts else None
                snap["gpu_mem_mb"] = int(parts[1].strip()) if len(parts) > 1 else None
        except Exception:
            pass
        return snap

    def _check(self, snap: dict):
        cpu = snap.get("cpu_pct", 0)
        disk = snap.get("disk_free_gb", 999)
        mem = snap.get("mem_avail_gb", 999)
        if cpu and cpu > 95:
            self._warn(f"CPU使用率 {cpu}% — 接近饱和")
        if disk < 10:
            self._warn(f"磁盘剩余 {disk:.1f}GB — 空间不足")
        if mem < 2:
            self._warn(f"可用内存 {mem:.1f}GB — 内存不足")

    def _warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  ⚠️ [ResourceMonitor] {msg}", flush=True)


# ═══════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════

def _hash_dict(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

def _hash_file(path: str) -> str:
    if not os.path.exists(path): return "?"
    with open(path, "rb") as f:
        return hashlib.md5(f.read(4096)).hexdigest()[:12]

def _short_ts() -> str:
    return datetime.now().strftime("%H%M%S")

def _run_cmd(cmd: str):
    import subprocess as sp
    r = sp.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败: {cmd}\n{r.stderr[:500]}")
