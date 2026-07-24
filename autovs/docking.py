"""Autovs 统一对接引擎模块

支持三引擎:
- smina:  CPU快速对接，不需要GPU
- gnina:  GPU对接，CNN打分，大多数靶点更准确
- DiffDock: GPU对接，扩散模型，PPI靶点更准确

引擎自动选择规则:
- PPI靶点 → DiffDock
- 通用靶点 + GPU可用 → GNINA
- 无GPU → smina (fallback)
- 用户可在策略中显式指定 engine 参数覆盖自动选择
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# ── Key type constants for smart engine selection ──────────────────────

PPI_TARGET_KEYWORDS = {
    "ppi", "protein-protein interaction", "protein protein",
    "dimer", "dimerization", "heterodimer", "homodimer",
    "complex interface", "binding partner", "oligomer",
    "bcl-2", "bcl2", "bcl-xl", "bcl2l1", "mcl-1", "mcl1",
    "bcl-w", "bcl2l2", "bfl-1", "bcl2a1", "bim", "bad", "bid",
    "bak", "bax", "noxa", "puma", "bcl-2 family",
    "mdm2", "mdmx", "p53", "tp53",
    "iap", "xiap", "ciap", "survivin", "birc",
    "smac", "diablo", "caspase",
    "il-", "tnf", "tnfr", "trail", "fas", "cd40", "cd95",
    "brd", "bet", "bromodomain",
    "ras", "raf", "kras", "nras", "hras",
    "pdz", "sh2", "sh3", "ww domain", "ph domain",
    "14-3-3", "calmodulin",
}

NON_PPI_TARGET_KEYWORDS = {
    "kinase", "tyrosine kinase", "serine/threonine kinase",
    "receptor", "gpcr", "g protein-coupled",
    "ion channel", "transporter",
    "protease", "enzyme", "hydrolase", "oxidoreductase",
    "nuclear receptor", "transcription factor",
    "phosphatase", "dehydrogenase", "isomerase", "transferase",
    "pde", "cox", "ace", "hiv", "rt", "integrase",
}


def detect_target_type(research: dict[str, Any] | None) -> str:
    """从靶点调研报告中检测靶点类型: 'ppi' | 'enzyme' | 'general'.

    返回值用于引擎选择:
    - ppi → DiffDock
    - enzyme → GNINA (高精度CNN)
    - general → GNINA (fallback)
    """
    if not research:
        return "general"

    text = json.dumps(research, ensure_ascii=False).lower()

    # 检查身份/功能字段
    identity = research.get("identity", {})
    function_text = str(identity.get("function", "")).lower()
    gene_symbol = str(identity.get("gene_symbol", "")).lower()
    uniprot_id = str(research.get("target_uniprot_id", "")).lower()

    combined = f"{text} {function_text} {gene_symbol} {uniprot_id}"

    # 首先检查明确的PPI标记
    ppi_score = 0
    for kw in PPI_TARGET_KEYWORDS:
        if kw in combined:
            ppi_score += 1

    # 检查非PPI标记
    non_ppi_score = 0
    for kw in NON_PPI_TARGET_KEYWORDS:
        if kw in combined:
            non_ppi_score += 1

    if ppi_score > non_ppi_score:
        return "ppi"
    elif non_ppi_score > 0:
        return "enzyme"  # 广义酶类/受体
    return "general"


def select_docking_engine(
    strategy_params: dict[str, Any] | None = None,
    research: dict[str, Any] | None = None,
    gpu_available: bool = False,
    cpu_only: bool = False,
) -> str:
    """选择合适的对接引擎.

    Args:
        strategy_params: 策略参数 (可包含 engine 字段显式指定)
        research: 靶点调研报告
        gpu_available: GPU是否可用
        cpu_only: 是否强制CPU模式

    Returns:
        'smina' | 'gnina' | 'diffdock'
    """
    # 1. 用户/策略显式指定
    if strategy_params:
        engine = str(strategy_params.get("engine", "")).lower().strip()
        if engine in {"smina", "gnina", "diffdock"}:
            return engine

    # 2. CPU-only强制smina
    if cpu_only:
        return "smina"

    # 3. 根据靶点类型选择
    target_type = detect_target_type(research)

    if target_type == "ppi" and gpu_available:
        return "diffdock"

    if gpu_available:
        return "gnina"

    # 4. Fallback
    return "smina"


# ── GNINA Slurm 提交 ──────────────────────────────────────────────────

def _build_gnina_slurm_script(
    gnina_bin: str,
    cmd: str,
    output_dir: Path,
    gpu_config: dict[str, Any],
    job_name: str = "autovs_gnina",
) -> Path:
    """构建 GNINA Slurm 提交脚本."""
    script = output_dir / "slurm_gnina.sh"
    gres = gpu_config.get("gres", "gpu:a100_2g.20gb:1")
    cpus = gpu_config.get("cpus", 10)
    mem = gpu_config.get("memory", "40G")
    walltime = gpu_config.get("time", "1-12:00:00")
    partition = gpu_config.get("partition", "gpu_long")

    script.write_text(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err

echo "[$(date)] Starting GNINA docking on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

{cmd}

EXIT_CODE=$?
echo "[$(date)] GNINA finished with exit code $EXIT_CODE"
exit $EXIT_CODE
""")
    script.chmod(0o755)
    return script


def submit_gnina_docking(
    receptor_pdbqt: Path,
    ligands_sdf: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    output_dir: Path,
    exhaustiveness: int = 8,
    num_modes: int = 5,
    cnn_scoring: str = "rescore",
    cnn_rotation: int = 1,
    seed: int = 61453,
    gnina_bin: str | None = None,
    gpu_config: dict[str, Any] | None = None,
    submit_slurm: bool = True,
) -> dict[str, Any]:
    """提交 GNINA 对接 (GPU).

    Returns:
        {slurm_job_id, output_dir, output_sdf, command, log_file, engine: 'gnina'}
    """
    gnina = gnina_bin or "gnina"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_sdf = output_dir / "gnina_poses.sdf"
    log_file = output_dir / "gnina.log"

    cmd_parts = [
        str(gnina),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligands_sdf),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size[0]),
        "--size_y", str(size[1]),
        "--size_z", str(size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--cnn_scoring", cnn_scoring,
        "--cnn_rotation", str(cnn_rotation),
        "--seed", str(seed),
        "--out", str(output_sdf),
        "--log", str(log_file),
    ]
    cmd = " ".join(cmd_parts)

    if submit_slurm:
        gpu_cfg = gpu_config or {}
        gpu_cfg.setdefault("gres", "gpu:a100_2g.20gb:1")
        gpu_cfg.setdefault("cpus", 10)
        gpu_cfg.setdefault("memory", "40G")
        gpu_cfg.setdefault("time", "1-12:00:00")
        script = _build_gnina_slurm_script(gnina, cmd, output_dir, gpu_cfg)
        result = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, timeout=30,
        )
        job_id = ""
        if result.returncode == 0 and result.stdout.strip():
            # 解析 "Submitted batch job 12345"
            parts = result.stdout.strip().split()
            job_id = parts[-1] if parts else ""
        return {
            "slurm_job_id": job_id,
            "output_dir": str(output_dir),
            "output_sdf": str(output_sdf),
            "log_file": str(log_file),
            "command": cmd,
            "engine": "gnina",
        }
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=86400)
        if result.returncode != 0:
            raise RuntimeError(f"GNINA failed: {result.stderr[-500:]}")
        return {
            "slurm_job_id": "",
            "output_dir": str(output_dir),
            "output_sdf": str(output_sdf),
            "log_file": str(log_file),
            "command": cmd,
            "engine": "gnina",
        }


# ── DiffDock Slurm 提交 ────────────────────────────────────────────────

def _build_diffdock_slurm_script(
    conda_env: str,
    diffdock_home: str,
    python_cmd: str,
    output_dir: Path,
    gpu_config: dict[str, Any],
    job_name: str = "autovs_diffdock",
) -> Path:
    """构建 DiffDock Slurm 提交脚本."""
    script = output_dir / "slurm_diffdock.sh"
    gres = gpu_config.get("gres", "gpu:a100_2g.20gb:1")
    cpus = gpu_config.get("cpus", 10)
    mem = gpu_config.get("memory", "40G")
    walltime = gpu_config.get("time", "1-12:00:00")
    partition = gpu_config.get("partition", "gpu_long")

    script.write_text(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err

echo "[$(date)] Starting DiffDock on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

export DIFFDOCK_HOME={diffdock_home}
source /users_home/wangpengzheng/miniforge3/bin/activate {conda_env}

echo "DiffDock Python: $(which python)"
echo "DiffDock Home: $DIFFDOCK_HOME"

cd {output_dir}
{python_cmd}

EXIT_CODE=$?
echo "[$(date)] DiffDock finished with exit code $EXIT_CODE"
exit $EXIT_CODE
""")
    script.chmod(0o755)
    return script


def submit_diffdock_docking(
    receptor_pdb: Path,
    ligands_smi: str,  # SMILES string or SDF path
    output_dir: Path,
    samples_per_complex: int = 10,
    inference_steps: int = 20,
    conda_env: str = "diffdock",
    diffdock_home: str = "/users_home/wangpengzheng/software/DiffDock",
    gpu_config: dict[str, Any] | None = None,
    submit_slurm: bool = True,
) -> dict[str, Any]:
    """提交 DiffDock 对接 (GPU, PPI靶点推荐).

    DiffDock需要每个配体独立运行。对于多配体库，需逐个提交。
    单配体模式用于 top-N 精筛或单个分子评估。

    Returns:
        {slurm_job_id, output_dir, result_json, engine: 'diffdock'}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 Python 调用命令
    wrapper = "/users_home/wangpengzheng/药物筛选智能体/scripts/diffdock_wrapper.py"
    python_cmd = (
        f"python {wrapper} {receptor_pdb} '{ligands_smi}' {output_dir}"
        f" --samples {samples_per_complex} --steps {inference_steps} --device cuda"
    )

    if submit_slurm:
        gpu_cfg = gpu_config or {}
        gpu_cfg.setdefault("gres", "gpu:a100_2g.20gb:1")
        gpu_cfg.setdefault("cpus", 10)
        gpu_cfg.setdefault("memory", "40G")
        gpu_cfg.setdefault("time", "1-12:00:00")
        script = _build_diffdock_slurm_script(
            conda_env, diffdock_home, python_cmd, output_dir, gpu_cfg,
        )
        result = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, timeout=30,
        )
        job_id = ""
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            job_id = parts[-1] if parts else ""
        return {
            "slurm_job_id": job_id,
            "output_dir": str(output_dir),
            "result_json": str(output_dir / "result.json"),
            "command": python_cmd,
            "engine": "diffdock",
        }
    else:
        env = os.environ.copy()
        env["DIFFDOCK_HOME"] = diffdock_home
        result = subprocess.run(
            python_cmd, shell=True, capture_output=True, text=True,
            timeout=86400, env=env, cwd=str(output_dir),
        )
        if result.returncode != 0:
            raise RuntimeError(f"DiffDock failed: {result.stderr[-500:]}")
        result_json = output_dir / "result.json"
        return {
            "slurm_job_id": "",
            "output_dir": str(output_dir),
            "result_json": str(result_json) if result_json.is_file() else "",
            "command": python_cmd,
            "engine": "diffdock",
        }


# ── 多样性选择 (Murcko Scaffold) ───────────────────────────────────────

def select_diverse_top_n(
    scores_csv: Path,
    output_csv: Path,
    top_n: int = 20,
    max_per_scaffold: int = 2,
    manifest_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """基于 Murcko 骨架多样性选择 Top-N 分子.

    从已对接的分子中按打分排序，限制每个骨架的最大数量，
    确保输出候选物覆盖多样化的化学空间。

    Args:
        scores_csv: 对接打分 CSV (需含 source_id, docking_affinity 列)
        output_csv: 输出 CSV 路径
        top_n: 选择总数
        max_per_scaffold: 每个骨架最多保留分子数
        manifest_csv: 分子清单 CSV (含 scaffold 列)

    Returns:
        选中的分子列表
    """
    from collections import defaultdict

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # 读取打分
    with scores_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # 补充 scaffold
    scaffold_map: dict[str, str] = {}
    if manifest_csv and manifest_csv.is_file():
        with manifest_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sid = row.get("source_id", "")
                scaf = row.get("scaffold", "")
                if sid and scaf:
                    scaffold_map[sid] = scaf

    # 为每个分子计算 scaffold (如果 manifest 中没有)
    for row in rows:
        sid = row.get("source_id", "")
        if sid not in scaffold_map:
            smiles = row.get("smiles", "")
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        scaffold_map[sid] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                    else:
                        scaffold_map[sid] = f"__invalid_{sid}"
                except Exception:
                    scaffold_map[sid] = f"__error_{sid}"
            else:
                scaffold_map[sid] = f"__nosmiles_{sid}"

    # 按 docking_affinity 排序 (从小到大, 亲和力更好)
    def _affinity(row: dict) -> float:
        try:
            return float(row.get("docking_affinity", 0))
        except (ValueError, TypeError):
            return 0.0

    rows.sort(key=_affinity)

    # 多样性选择
    selected: list[dict] = []
    scaffold_counts = defaultdict(int)

    for row in rows:
        sid = row.get("source_id", "")
        scaffold = scaffold_map.get(sid, f"__unknown_{sid}")
        if scaffold_counts[scaffold] >= max_per_scaffold:
            continue
        scaffold_counts[scaffold] += 1
        row["rank"] = len(selected) + 1
        selected.append(row)
        if len(selected) >= top_n:
            break

    # 写入输出
    if selected:
        fields = list(selected[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)

    return selected


# ── 对接结果解析 ──────────────────────────────────────────────────────

def parse_docking_scores(
    poses_sdf: Path,
    manifest_csv: Path | None = None,
    engine: str = "smina",
) -> Path:
    """解析对接输出 SDF，生成标准化打分 CSV.

    Args:
        poses_sdf: 对接输出 SDF 路径
        manifest_csv: 分子清单 CSV
        engine: 'smina' | 'gnina' | 'diffdock'

    Returns:
        生成的 scores CSV 路径
    """
    from rdkit import Chem

    manifest_rows: dict[str, dict] = {}
    if manifest_csv and manifest_csv.is_file():
        with manifest_csv.open(encoding="utf-8-sig", newline="") as handle:
            manifest_rows = {
                row["source_id"]: row for row in csv.DictReader(handle)
            }

    best: dict[str, dict] = {}
    supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False, strictParsing=False)

    for mol in supplier:
        if mol is None:
            continue

        try:
            source_id = (
                mol.GetProp("source_id") if mol.HasProp("source_id")
                else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            )
        except (RuntimeError, KeyError):
            continue

        # 提取亲和力 (不同引擎使用不同属性)
        affinity = None
        cnn_score = None
        cnn_affinity = None
        cnn_vs = None
        confidence = None

        if engine == "gnina":
            for prop, target in [
                ("CNNscore", "cnn_score"), ("CNNaffinity", "cnn_affinity"),
                ("CNN_VS", "cnn_vs"),
            ]:
                if mol.HasProp(prop):
                    try:
                        val = float(mol.GetProp(prop))
                        if target == "cnn_score":
                            cnn_score = val
                        elif target == "cnn_affinity":
                            cnn_affinity = val
                        elif target == "cnn_vs":
                            cnn_vs = val
                    except ValueError:
                        pass
            # GNINA also has minimizedAffinity
            for prop in ("minimizedAffinity", "affinity"):
                if mol.HasProp(prop):
                    try:
                        affinity = float(mol.GetProp(prop))
                        break
                    except ValueError:
                        pass
        elif engine == "diffdock":
            if mol.HasProp("confidence"):
                try:
                    confidence = float(mol.GetProp("confidence"))
                except ValueError:
                    pass
            # DiffDock doesn't produce traditional affinity scores
            affinity = confidence
        else:  # smina
            for prop in ("minimizedAffinity", "affinity", "SCORE"):
                if mol.HasProp(prop):
                    try:
                        affinity = float(mol.GetProp(prop))
                        break
                    except ValueError:
                        pass

        if affinity is None:
            continue

        row = dict(manifest_rows.get(source_id, {}))
        smiles = row.get("smiles", "")
        if not smiles:
            try:
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smiles = ""

        row.update({
            "source_id": source_id,
            "smiles": smiles,
            "docking_affinity": affinity,
        })
        if cnn_score is not None:
            row["cnn_score"] = cnn_score
        if cnn_affinity is not None:
            row["cnn_affinity"] = cnn_affinity
        if cnn_vs is not None:
            row["cnn_vs"] = cnn_vs
        if confidence is not None:
            row["diffdock_confidence"] = confidence

        # 保留最佳得分姿态
        if engine == "diffdock":
            # DiffDock: 选择最高置信度
            existing_conf = float(best.get(source_id, {}).get("diffdock_confidence", -999))
            if confidence is not None and confidence > existing_conf:
                best[source_id] = row
        elif engine == "gnina":
            # GNINA: 优先 CNN_VS，其次 CNNaffinity，最后 docking_affinity
            existing_cnn_vs = float(best.get(source_id, {}).get("cnn_vs", -999))
            if cnn_vs is not None and cnn_vs > existing_cnn_vs:
                best[source_id] = row
            elif source_id not in best:
                best[source_id] = row
        else:
            # smina: 最佳 affinity (最低值)
            existing_aff = float(best.get(source_id, {}).get("docking_affinity", 999))
            if affinity < existing_aff:
                best[source_id] = row

    score_rows = list(best.values())
    if not score_rows:
        raise RuntimeError(f"{engine} output contains no parseable molecule scores")

    # 收集所有字段
    all_fields = list(dict.fromkeys(
        key for row in score_rows for key in row
    ))
    scores_csv = poses_sdf.parent / f"{engine}_scores.csv"
    with scores_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(score_rows)

    return scores_csv
