"""Test 01: GNINA GPU Slurm对接验证.

用法:
  # 登录节点提交模式(提交到Slurm)
  python tests/phase2_validation/test_01_gnina_gpu.py

  # Slurm作业内部模式(由Slurm脚本调用)
  python tests/phase2_validation/test_01_gnina_gpu.py --local
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autovs.docking import submit_gnina_docking, parse_docking_scores

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_baseline_inputs() -> dict[str, Path] | None:
    """获取已有baseline任务的input文件."""
    tasks_root = PROJECT_ROOT / "runtime" / "tasks"
    if not tasks_root.is_dir():
        return None
    for task_dir in sorted(tasks_root.iterdir(), reverse=True):
        steps = task_dir / "steps"
        receptor = steps / "protein-preparation" / "receptor.pdbqt"
        ligands = None
        for p in [
            steps / "molecule-standardization" / "prepared_library.sdf",
            steps / "physicochemical-filtering" / "prepared_library.sdf",
            steps / "format-conversion" / "converted.sdf",
        ]:
            if p.is_file():
                ligands = p
                break
        manifest = None
        for p in [
            steps / "molecule-standardization" / "manifest.csv",
            steps / "physicochemical-filtering" / "manifest.csv",
        ]:
            if p.is_file():
                manifest = p
                break

        if receptor.is_file() and ligands:
            # 口袋参数
            pocket_json = steps / "pocket-definition" / "pocket.json"
            center = (-15.36, 2.24, -9.56)
            size = (24, 24, 24)
            if pocket_json.is_file():
                try:
                    data = json.loads(pocket_json.read_text())
                    pkt = data.get("selected_pocket", {})
                    center = tuple(pkt.get("center", center))
                    size = tuple(pkt.get("size", size))
                except (json.JSONDecodeError, KeyError):
                    pass

            return {
                "receptor_pdbqt": receptor,
                "ligands_sdf": ligands,
                "manifest_csv": manifest,
                "center": center,
                "size": size,
            }
    return None


def run_gnina_local(inputs: dict[str, Path], work_dir: Path) -> dict:
    """在Slurm作业内本地运行GNINA对接和验证."""
    from rdkit import Chem

    print(f"GNINA local run: {work_dir}", flush=True)

    result = submit_gnina_docking(
        receptor_pdbqt=inputs["receptor_pdbqt"],
        ligands_sdf=inputs["ligands_sdf"],
        center=inputs["center"],
        size=inputs["size"],
        output_dir=work_dir,
        exhaustiveness=8,
        num_modes=5,
        cnn_scoring="rescore",
        cnn_rotation=1,
        submit_slurm=False,
        gnina_bin="/users_home/wangpengzheng/software/gnina",
    )

    output_sdf = Path(result["output_sdf"])
    if not output_sdf.is_file():
        return {"status": "failed", "error": "GNINA output SDF not found"}

    # 解析结果
    manifest = inputs.get("manifest_csv")
    try:
        scores_csv = parse_docking_scores(output_sdf, manifest, engine="gnina")
    except Exception as exc:
        return {"status": "failed", "error": f"parse_docking_scores: {exc}"}

    # 验证CNN属性
    checks = []
    mol_count = 0
    has_cnn_score = False
    has_cnn_affinity = False
    has_cnn_vs = False

    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False, strictParsing=False)
    cnn_vs_values = []
    for mol in supplier:
        if mol is None:
            continue
        mol_count += 1
        for prop in ("CNNscore",):
            if mol.HasProp(prop):
                has_cnn_score = True
                break
        for prop in ("CNNaffinity",):
            if mol.HasProp(prop):
                has_cnn_affinity = True
        if mol.HasProp("CNN_VS"):
            try:
                cnn_vs_values.append(float(mol.GetProp("CNN_VS")))
            except ValueError:
                pass
            has_cnn_vs = True

    checks.append({"check": "GNINA输出有分子", "pass": mol_count > 0, "detail": f"{mol_count} poses"})
    checks.append({"check": "CNNscore存在", "pass": has_cnn_score, "detail": ""})
    checks.append({"check": "CNNaffinity存在", "pass": has_cnn_affinity, "detail": ""})
    checks.append({"check": "CNN_VS存在", "pass": has_cnn_vs, "detail": f"top={max(cnn_vs_values):.4f}" if cnn_vs_values else ""})

    # 读取打分CSV
    mols_in_csv = 0
    if scores_csv.is_file():
        with scores_csv.open(encoding="utf-8-sig", newline="") as f:
            mols_in_csv = sum(1 for _ in csv.DictReader(f))
    checks.append({"check": "打分CSV有数据", "pass": mols_in_csv > 0, "detail": f"{mols_in_csv} mols"})

    all_ok = all(c["pass"] for c in checks)
    return {
        "status": "completed" if all_ok else "partial",
        "all_ok": all_ok,
        "poses_count": mol_count,
        "mols_in_csv": mols_in_csv,
        "top_cnn_vs": max(cnn_vs_values) if cnn_vs_values else None,
        "checks": checks,
        "scores_csv": str(scores_csv),
        "output_sdf": str(output_sdf),
    }


def run_validation() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run in Slurm job (no sbatch)")
    args = parser.parse_args()

    if args.local:
        # Slurm作业内部模式
        inputs = _get_baseline_inputs()
        if inputs is None:
            print("❌ 未找到baseline输入文件")
            return 1
        work_dir = Path("/tmp/autovs_gnina_test")
        work_dir.mkdir(parents=True, exist_ok=True)
        result = run_gnina_local(inputs, work_dir)
        report_path = REPORT_DIR / "gnina_result.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n报告: {report_path}")
        for c in result.get("checks", []):
            icon = "✅" if c["pass"] else "❌"
            print(f"  {icon} {c['check']}: {c['detail']}")
        return 0 if result.get("all_ok") else 1

    # 登录节点提交模式
    inputs = _get_baseline_inputs()
    if inputs is None:
        print("❌ 未找到baseline输入文件，需要先运行一次完整流水线")
        return 1

    print(f"\n{'='*60}")
    print(f"  GNINA GPU Slurm 对接验证")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  受体: {inputs['receptor_pdbqt']}")
    print(f"  配体: {inputs['ligands_sdf']}")
    print(f"  口袋中心: {inputs['center']}")
    print(f"  口袋大小: {inputs['size']}")

    # 构建Slurm脚本
    script = _build_slurm_wrapper()
    script_path = REPORT_DIR / "slurm_gnina_test.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    # 提交
    print(f"\n  提交Slurm作业...")
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  ❌ sbatch失败: {result.stderr}")
        return 1

    job_id = result.stdout.strip().split()[-1]
    print(f"  ✅ Slurm作业提交: {job_id}")
    print(f"  📝 轮询状态 (最长6小时)...")

    # 轮询
    deadline = time.time() + 21600
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
            print(f"\n  ⚠️ 作业不在队列中，假设完成")
            break
        elapsed = int(time.time() - (deadline - 21600))
        print(f"  ⏳ 作业状态: {state} (已等待 {elapsed}s)", end="\r", flush=True)
        time.sleep(30)

    # 读取结果
    result_path = REPORT_DIR / "gnina_result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text())
        print(f"\n{'='*60}")
        for c in data.get("checks", []):
            icon = "✅" if c["pass"] else "❌"
            print(f"  {icon} {c['check']}: {c['detail']}")
        all_ok = data.get("all_ok", False)
        print(f"  {'✅ GNINA验证通过' if all_ok else '❌ GNINA验证失败'}")
        return 0 if all_ok else 1
    else:
        print(f"  ⚠️ 结果文件未生成")
        return 1


def _build_slurm_wrapper() -> str:
    """构建GNINA测试的Slurm提交脚本."""
    test_script = str(Path(__file__).resolve())
    report_dir = str(REPORT_DIR)
    return f"""#!/bin/bash
#SBATCH --job-name=autovs_gnina_test
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output={report_dir}/gnina_slurm_%j.out
#SBATCH --error={report_dir}/gnina_slurm_%j.err

echo "[$(date)] GNINA test starting on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd {PROJECT_ROOT}
/users_home/wangpengzheng/miniforge3/bin/python {test_script} --local

EXIT_CODE=$?
echo "[$(date)] GNINA test finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""


if __name__ == "__main__":
    sys.exit(run_validation())
