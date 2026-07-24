"""Test 00: 配置正确性验证 — 无需 Slurm，登录节点直接运行."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = Path(__file__).resolve().parent


def check_executable(name: str, path: str) -> dict:
    """检查二进制是否可执行."""
    p = Path(path)
    ok = p.is_file() and (p.stat().st_mode & 0o100 != 0)
    return {"name": name, "path": str(p), "executable": ok, "reason": "" if ok else f"{path} not found or not executable"}


def check_slurm_binary(name: str) -> dict:
    """检查 Slurm 命令是否可用."""
    try:
        result = subprocess.run(["which", name], capture_output=True, text=True, timeout=10)
        ok = result.returncode == 0
        return {"name": name, "path": result.stdout.strip(), "available": ok}
    except Exception as e:
        return {"name": name, "path": "", "available": False, "error": str(e)}


def check_conda_env(env_name: str) -> dict:
    """检查 conda 环境是否存在."""
    env_path = Path(f"/users_home/wangpengzheng/miniforge3/envs/{env_name}")
    ok = env_path.is_dir()
    return {"name": env_name, "path": str(env_path), "exists": ok}


def check_container(path: str) -> dict:
    """检查 Apptainer 容器."""
    p = PROJECT_ROOT / path
    ok = p.is_file() and p.stat().st_size > 0
    return {"name": path, "path": str(p), "exists": ok, "size_mb": p.stat().st_size / (1024 * 1024) if ok else 0}


def run_validation() -> int:
    """运行配置验证."""
    checks = []
    passed = 0

    # 1. Slurm 工具
    for name in ["sbatch", "squeue", "scancel"]:
        c = check_slurm_binary(name)
        checks.append({"check": f"Slurm {name}", "pass": c["available"], "detail": c["path"]})
        if c["available"]:
            passed += 1

    # 2. 对接引擎
    for name, path in [
        ("smina", "/users_home/wangpengzheng/miniforge3/envs/smina_stage2/bin/smina"),
        ("gnina", "/users_home/wangpengzheng/software/gnina"),
    ]:
        c = check_executable(name, path)
        checks.append({"check": name, "pass": c["executable"], "detail": c["path"]})
        if c["executable"]:
            passed += 1

    # 3. Conda 环境
    for env_name in ["diffdock", "plip", "smina_stage2"]:
        c = check_conda_env(env_name)
        checks.append({"check": f"conda env {env_name}", "pass": c["exists"], "detail": c["path"]})
        if c["exists"]:
            passed += 1

    # 4. DiffDock 模型
    for model_name in [
        "score_model/best_ema_inference_epoch_model.pt",
        "confidence_model/best_model_epoch75.pt",
    ]:
        p = Path(f"/users_home/wangpengzheng/software/DiffDock/workdir/v1.1/{model_name}")
        ok = p.is_file()
        checks.append({"check": f"DiffDock {model_name.split('/')[0]}", "pass": ok, "detail": str(p)})
        if ok:
            passed += 1

    # 5. GROMACS 容器
    c = check_container("containers/gromacs_md.sif")
    checks.append({"check": "GROMACS container", "pass": c["exists"], "detail": f"{c['size_mb']:.0f} MB"})
    if c["exists"]:
        passed += 1

    # 6. 配置检查
    import tomllib
    try:
        with open(PROJECT_ROOT / "config" / "tools.toml", "rb") as f:
            raw = tomllib.load(f)
        gpu = raw.get("slurm", {}).get("gpu", {})
        cpu = raw.get("slurm", {}).get("cpu", {})
        gres = gpu.get("gres", "")
        cpu_mem = cpu.get("memory", "")
        gpu_ok = "40gb" in gres.lower() or "3g" in gres.lower()
        cpu_ok = "64" in str(cpu_mem)
        checks.append({"check": "GPU gres 40GB", "pass": gpu_ok, "detail": gres})
        checks.append({"check": "CPU memory 64GB", "pass": cpu_ok, "detail": cpu_mem})
        if gpu_ok:
            passed += 1
        if cpu_ok:
            passed += 1
    except Exception as e:
        checks.append({"check": "config/tools.toml", "pass": False, "detail": str(e)})

    # 7. Slurm 分区
    try:
        result = subprocess.run(["sinfo", "-o", "%P", "-h", "--noheader"], capture_output=True, text=True, timeout=10)
        partitions = set(result.stdout.strip().split())
        for pname in ["gpu_long", "cpu_only"]:
            ok = pname in partitions
            checks.append({"check": f"Slurm partition {pname}", "pass": ok, "detail": "available" if ok else "not found"})
            if ok:
                passed += 1
    except Exception as e:
        checks.append({"check": "Slurm sinfo", "pass": False, "detail": str(e)})

    total = len(checks)
    print(f"\n{'='*60}")
    print(f"  Phase 2 配置验证")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    for c in checks:
        icon = "✅" if c["pass"] else "❌"
        print(f"  {icon} {c['check']:<35} {c['detail'][:50]}")
    print(f"  {'='*60}")
    print(f"  通过: {passed}/{total}")
    all_ok = passed == total
    print(f"  {'✅ 环境就绪' if all_ok else '❌ 环境存在问题'}")

    report = {
        "test": "config_validation",
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "all_ok": all_ok,
        "checks": checks,
    }
    report_path = REPORT_DIR / "config_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告: {report_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_validation())
