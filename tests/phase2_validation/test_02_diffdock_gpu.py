"""Test 02: DiffDock GPU Slurm PPI对接验证.

用法:
  python tests/phase2_validation/test_02_diffdock_gpu.py           # 提交到Slurm
  python tests/phase2_validation/test_02_diffdock_gpu.py --local   # Slurm内部运行
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIFFDOCK_HOME = "/users_home/wangpengzheng/software/DiffDock"
DIFFDOCK_PYTHON = "/users_home/wangpengzheng/miniforge3/envs/diffdock/bin/python"

# 测试配体: BCL-2 相关药物和已知PPI抑制剂
TEST_LIGANDS = [
    ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc2ncccn2"),
    ("Celecoxib", "Cc1nn(C)c(C)c1C(=O)NS(=O)(=O)c1ccc(C)cc1"),
    ("Dasatinib", "Cc1nc(Nc2cc(Cl)ccc2NC(=O)C=Cc2cccnc2)nc(N2CCN(CCO)CC2)n1"),
]


def run_diffdock_local(protein_path: Path, ligands: list[tuple[str, str]],
                       work_dir: Path) -> dict:
    """在Slurm作业内运行DiffDock."""
    import os as _os

    wrapper = PROJECT_ROOT / "scripts" / "diffdock_wrapper.py"
    if not wrapper.is_file():
        return {"status": "failed", "error": f"wrapper not found: {wrapper}"}

    results = []
    all_checks = []

    for name, smiles in ligands:
        out_dir = work_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"{DIFFDOCK_PYTHON} {wrapper} {protein_path} '{smiles}' {out_dir}"
            f" --samples 10 --steps 20 --device cuda"
        )

        print(f"\n  [{name}] 运行DiffDock...", flush=True)
        t0 = time.time()

        env = _os.environ.copy()
        env["DIFFDOCK_HOME"] = DIFFDOCK_HOME

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=7200, env=env, cwd=str(out_dir),
        )

        elapsed = time.time() - t0
        ok = result.returncode == 0

        result_json = out_dir / "result.json"
        poses_data = {}
        if result_json.is_file():
            try:
                poses_data = json.loads(result_json.read_text())
            except json.JSONDecodeError:
                pass

        poses_count = len(poses_data.get("poses", []))
        top_conf = poses_data.get("top_confidence")

        checks = [
            {"check": f"{name}: exit 0", "pass": ok, "detail": f"{elapsed:.0f}s"},
            {"check": f"{name}: result.json", "pass": result_json.is_file(), "detail": f"{poses_count} poses"},
            {"check": f"{name}: confidence > 0", "pass": top_conf is not None and top_conf > 0, "detail": str(top_conf)},
        ]
        results.append({
            "name": name,
            "success": ok,
            "elapsed": elapsed,
            "poses_count": poses_count,
            "top_confidence": top_conf,
            "checks": checks,
        })
        all_checks.extend(checks)

        for c in checks:
            icon = "✅" if c["pass"] else "❌"
            print(f"    {icon} {c['check']}: {c['detail']}", flush=True)

    all_ok = all(c["pass"] for c in all_checks)
    return {
        "status": "completed" if all_ok else "partial",
        "all_ok": all_ok,
        "ligands_tested": len(ligands),
        "results": results,
        "checks": all_checks,
    }


def run_validation() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run in Slurm job")
    parser.add_argument("--smiles", type=str, help="Override: SMILES string for single ligand")
    parser.add_argument("--name", type=str, default="test_ligand", help="Name for --smiles ligand")
    args = parser.parse_args()

    protein_path = PROJECT_ROOT / "demo" / "6O0K.pdb"
    if not protein_path.is_file():
        print(f"❌ 蛋白文件不存在: {protein_path}")
        return 1

    if args.local:
        ligands = TEST_LIGANDS
        if args.smiles:
            ligands = [(args.name, args.smiles)]
        work_dir = Path("/tmp/autovs_diffdock_test")
        work_dir.mkdir(parents=True, exist_ok=True)
        result = run_diffdock_local(protein_path, ligands, work_dir)
        report_path = REPORT_DIR / "diffdock_result.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"\n报告: {report_path}")
        return 0 if result.get("all_ok") else 1

    # 提交模式
    print(f"\n{'='*60}")
    print(f"  DiffDock GPU Slurm PPI对接验证")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  靶蛋白: {protein_path} (BCL-2 PPI靶点)")
    print(f"  配体数: {len(TEST_LIGANDS)} (Imatinib, Celecoxib, Dasatinib)")

    script = _build_slurm_wrapper()
    script_path = REPORT_DIR / "slurm_diffdock_test.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  ❌ sbatch失败: {result.stderr}")
        return 1

    job_id = result.stdout.strip().split()[-1]
    print(f"  ✅ 作业提交: {job_id}")

    # 轮询(最长12小时)
    deadline = time.time() + 43200
    while time.time() < deadline:
        r = subprocess.run(
            ["squeue", "-j", job_id, "-o", "%T", "-h", "--noheader"],
            capture_output=True, text=True, timeout=30,
        )
        state = r.stdout.strip()
        if state in ("COMPLETED",):
            print(f"\n  ✅ 作业完成!")
            break
        if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
            print(f"\n  ❌ 作业失败: {state}")
            return 1
        if not state:
            print(f"\n  ⚠️ 作业不在队列中")
            break
        elapsed = int(time.time() - (deadline - 43200))
        print(f"  ⏳ 状态: {state} ({elapsed}s)", end="\r", flush=True)
        time.sleep(30)

    result_path = REPORT_DIR / "diffdock_result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text())
        print(f"\n{'='*60}")
        for c in data.get("checks", []):
            icon = "✅" if c["pass"] else "❌"
            print(f"  {icon} {c['check']}: {c['detail']}")
        all_ok = data.get("all_ok", False)
        print(f"  {'✅ DiffDock验证通过' if all_ok else '❌ DiffDock验证失败'}")
        return 0 if all_ok else 1
    else:
        print(f"  ⚠️ 结果文件未生成")
        return 1


def _build_slurm_wrapper() -> str:
    test_script = str(Path(__file__).resolve())
    report_dir = str(REPORT_DIR)
    return f"""#!/bin/bash
#SBATCH --job-name=autovs_diffdock_test
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output={report_dir}/diffdock_slurm_%j.out
#SBATCH --error={report_dir}/diffdock_slurm_%j.err

echo "[$(date)] DiffDock test starting on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

export DIFFDOCK_HOME="{DIFFDOCK_HOME}"

echo "DiffDock Home: $DIFFDOCK_HOME"

cd {PROJECT_ROOT}
/users_home/wangpengzheng/miniforge3/bin/python {test_script} --local

EXIT_CODE=$?
echo "[$(date)] DiffDock test finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""


if __name__ == "__main__":
    sys.exit(run_validation())
