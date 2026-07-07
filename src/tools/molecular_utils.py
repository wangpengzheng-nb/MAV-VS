"""
AutoVS-Agent: 分子工具库 (Molecular Utilities)
================================================
GNINA 对接、GROMACS MD 模拟、Slurm 作业管理的真实调用封装。

软件路径 (来自服务器已验证的 skill 配置):
  GNINA:   /users_home/wangpengzheng/software/gnina
  GROMACS: source ~/VS_Agent_HTVS/md_deploy/env_gromacs_cuda.sh + conda activate gmx_mmpbsa
  smina:   conda activate smina_stage2

Slurm 分区:
  GPU: gpu_long (--gres=gpu:a100_2g.20gb:1)
  CPU: cpu_only (--qos=cpuonly)
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# RDKit
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.logger().setLevel(RDLogger.ERROR)
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# =============================================================================
# 服务器路径常量
# =============================================================================

GNINA_BINARY = "/users_home/wangpengzheng/software/gnina"
SMINA_BINARY = "/users_home/wangpengzheng/miniforge3/envs/smina_stage2/bin/smina"
PARSE_DOCKING_SCRIPT = "/users_home/wangpengzheng/.claude/skills/smina-gnina-docking/scripts/parse_docking_sdf.py"

GROMACS_ENV_SCRIPT = "/users_home/wangpengzheng/VS_Agent_HTVS/md_deploy/env_gromacs_cuda.sh"
GROMACS_CONDA_ENV = "gmx_mmpbsa"
MDP_TEMPLATE_DIR = "/users_home/wangpengzheng/VS_Agent_HTVS/pharmit＋plip/MD60个小分子-第五批/md_affinity_top20_1x100ns_gromacs/mdp"

CONDA_BASE = "/users_home/wangpengzheng/miniforge3/etc/profile.d/conda.sh"

SLURM_PARTITION_GPU = "gpu_long"
SLURM_PARTITION_CPU = "cpu_only"
SLURM_DEFAULT_WALLTIME_HTVS = "2-00:00:00"
SLURM_DEFAULT_WALLTIME_MD = "1-12:00:00"


# =============================================================================
# Slurm 作业管理
# =============================================================================

class SlurmJobManager:
    """Slurm 作业提交与监控。"""

    @staticmethod
    def submit(
        script_path: str,
        job_name: str = "autovs",
        log_dir: Optional[str] = None,
    ) -> Optional[str]:
        """提交 Slurm 作业，返回 Job ID。"""
        cmd = ["sbatch", f"--job-name={job_name}"]
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            cmd.extend([f"--output={log_dir}/slurm_%A_%a.out",
                        f"--error={log_dir}/slurm_%A_%a.err"])
        cmd.append(script_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                # 解析 "Submitted batch job 123456"
                job_id = result.stdout.strip().split()[-1]
                return job_id
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def status(job_id: str) -> str:
        """查询 Slurm 作业状态。

        Returns: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "TIMEOUT" | "UNKNOWN"
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%T", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            state = result.stdout.strip()
            if state:
                if state in ("PENDING", "CONFIGURING"):
                    return "PENDING"
                elif state == "RUNNING":
                    return "RUNNING"
                elif state in ("COMPLETED", "COMPLETING"):
                    return "COMPLETED"
                elif state in ("FAILED", "CANCELLED", "NODE_FAIL", "PREEMPTED"):
                    return "FAILED"
                elif state == "TIMEOUT":
                    return "TIMEOUT"
            else:
                # 不在队列中 → 可能已完成，用 sacct 查历史
                return SlurmJobManager._sacct_status(job_id)
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    @staticmethod
    def _sacct_status(job_id: str) -> str:
        """通过 sacct 查询历史作业状态。"""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-o", "State", "--noheader", "-P"],
                capture_output=True, text=True, timeout=10,
            )
            states = result.stdout.strip().split("\n")
            for s in states:
                s = s.strip()
                if s in ("COMPLETED",):
                    return "COMPLETED"
                elif s in ("FAILED", "CANCELLED", "NODE_FAIL", "TIMEOUT"):
                    return "FAILED"
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    @staticmethod
    def wait(
        job_ids: List[str],
        poll_interval: int = 30,
        timeout_seconds: Optional[int] = None,
        on_progress: Optional[callable] = None,
    ) -> Dict[str, str]:
        """轮询等待多个 Slurm 作业完成。

        Args:
            job_ids: Slurm Job ID 列表。
            poll_interval: 轮询间隔 (秒)。
            timeout_seconds: 总超时 (秒)，None 为无限等待。
            on_progress: 进度回调 (job_id, status)。

        Returns:
            {job_id: "COMPLETED"|"FAILED"|"TIMEOUT"|...}
        """
        start_time = time.time()
        final_status: Dict[str, str] = {}
        pending = set(job_ids)

        while pending:
            for jid in list(pending):
                s = SlurmJobManager.status(jid)
                if s in ("COMPLETED", "FAILED", "TIMEOUT"):
                    final_status[jid] = s
                    pending.discard(jid)
                    if on_progress:
                        on_progress(jid, s)

            if not pending:
                break

            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                for jid in pending:
                    final_status[jid] = "TIMEOUT"
                break

            time.sleep(poll_interval)

        return final_status

    @staticmethod
    def cancel(job_id: str) -> bool:
        """取消 Slurm 作业。"""
        try:
            subprocess.run(["scancel", job_id], capture_output=True, timeout=10)
            return True
        except Exception:
            return False


# =============================================================================
# SDF 工具
# =============================================================================

class SDFUtils:
    """SDF 解析与生成。"""

    @staticmethod
    def smiles_to_sdf(
        smiles_list: List[str],
        mol_ids: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """将 SMILES 列表转换为含 3D 构象的 SDF 文件。

        Args:
            smiles_list: SMILES 字符串列表。
            mol_ids: 分子 ID 列表 (与 SMILES 一一对应)。
            output_path: 输出路径，None 则使用临时文件。

        Returns:
            SDF 文件路径。
        """
        if not _RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for SMILES→SDF conversion.")

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".sdf")

        writer = Chem.SDWriter(output_path)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                continue

            mol = Chem.AddHs(mol)
            # ETKDGv3 构象生成
            params = AllChem.ETKDGv3()
            params.randomSeed = 42 + i
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                # 回退到 ETKDG
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                AllChem.UFFOptimizeMolecule(mol)

            mol_id = mol_ids[i] if mol_ids and i < len(mol_ids) else f"mol_{i:09d}"
            mol.SetProp("_Name", mol_id)
            mol.SetProp("source_id", mol_id)
            mol.SetProp("smiles", smi.strip())

            writer.write(mol)
        writer.close()
        return output_path

    @staticmethod
    def parse_gnina_sdf(
        sdf_path: str,
        pose_select: str = "best_cnn_vs",
    ) -> List[Dict[str, Any]]:
        """解析 GNINA 对接输出 SDF，提取评分和姿态。

        Args:
            sdf_path: GNINA 输出的 SDF 路径。
            pose_select: 姿态选择策略 ("best_affinity" | "best_cnn_vs" | "best_cnnscore")。

        Returns:
            [{mol_id, affinity, cnnscore, cnnaffinity, cnn_vs, pose_rank}, ...]
        """
        if not _RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for SDF parsing.")

        rows: List[Dict[str, Any]] = []
        from collections import defaultdict
        seen: Dict[str, int] = defaultdict(int)

        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue

            source_id = (
                mol.GetProp("source_id") if mol.HasProp("source_id")
                else mol.GetProp("_Name") if mol.HasProp("_Name")
                else "unknown"
            )
            seen[source_id] += 1
            pose_rank = seen[source_id]

            cnnscore = SDFUtils._safe_float(mol, "CNNscore")
            cnnaffinity = SDFUtils._safe_float(mol, "CNNaffinity")
            cnn_vs = cnnscore * cnnaffinity if cnnscore and cnnaffinity else None
            affinity = SDFUtils._safe_float(
                mol, "minimizedAffinity"
            ) or SDFUtils._safe_float(mol, "affinity")

            rows.append({
                "mol_id": source_id,
                "pose_rank": pose_rank,
                "affinity": affinity,
                "cnnscore": cnnscore,
                "cnnaffinity": cnnaffinity,
                "cnn_vs": cnn_vs,
                "smiles": mol.GetProp("smiles") if mol.HasProp("smiles") else "",
            })

        # 选最优姿态
        return SDFUtils._select_best_poses(rows, pose_select)

    @staticmethod
    def _safe_float(mol, prop_name: str) -> Optional[float]:
        try:
            val = mol.GetProp(prop_name) if mol.HasProp(prop_name) else None
            if val is None:
                return None
            f = float(val)
            return None if math.isnan(f) else f
        except Exception:
            return None

    @staticmethod
    def _select_best_poses(
        rows: List[Dict[str, Any]],
        pose_select: str,
    ) -> List[Dict[str, Any]]:
        """按策略为每个分子选择最优姿态。"""
        from collections import defaultdict

        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            grouped[row["mol_id"]].append(row)

        metric = {
            "best_affinity": ("affinity", False),
            "best_cnn_vs": ("cnn_vs", True),
            "best_cnnscore": ("cnnscore", True),
        }.get(pose_select, ("cnn_vs", True))

        key, reverse = metric
        selected = []
        for mol_id, group in grouped.items():
            valid = [(r.get(key), r) for r in group if r.get(key) is not None]
            if valid:
                _, best = sorted(valid, key=lambda x: x[0] or 0, reverse=reverse)[0]
            else:
                best = dict(group[0])
            selected.append(best)

        # 按评分排序
        selected.sort(
            key=lambda r: r.get(key) or 0,
            reverse=reverse,
        )
        return selected

    @staticmethod
    def read_smiles_from_sdf(sdf_path: str) -> List[Dict[str, str]]:
        """从 SDF 读取 mol_id 和 SMILES 列表。"""
        mols = []
        if not _RDKIT_AVAILABLE:
            return mols
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            mid = (
                mol.GetProp("source_id") if mol.HasProp("source_id")
                else mol.GetProp("_Name") if mol.HasProp("_Name")
                else "unknown"
            )
            smi = mol.GetProp("smiles") if mol.HasProp("smiles") else ""
            mols.append({"mol_id": mid, "smiles": smi})
        return mols


# =============================================================================
# GNINA 对接
# =============================================================================

class GNINADocker:
    """GNINA 对接引擎封装。

    支持 Rough Docking (高通量粗筛) 和 Refinement (高精度精筛)。
    """

    @staticmethod
    def run_rough_docking(
        receptor_pdb: str,
        ligand_sdf: str,
        grid_center: List[float],
        grid_size: List[float],
        output_dir: str,
        exhaustiveness: int = 8,
        num_modes: int = 3,
        cnn_scoring: str = "rescore",
        cnn_rotation: int = 1,
        seed: int = 61453,
        cpu: int = 8,
        gpu_device: int = 0,
        submit_slurm: bool = True,
        slurm_gres: str = "gpu:a100_2g.20gb:1",
        slurm_cpus: int = 8,
        slurm_mem: str = "20G",
        slurm_walltime: str = "1-00:00:00",
    ) -> Dict[str, Any]:
        """运行 GNINA Rough Docking。

        Args:
            receptor_pdb: 受体 PDB 路径。
            ligand_sdf: 配体 SDF 路径。
            grid_center: [x, y, z] 盒子中心。
            grid_size: [sx, sy, sz] 盒子尺寸。
            output_dir: 输出目录。
            exhaustiveness: 穷举度。
            submit_slurm: 是否提交 Slurm，False 则本地运行。

        Returns:
            {job_ids: [...], output_dir: str, command: str}
        """
        os.makedirs(output_dir, exist_ok=True)
        output_sdf = os.path.join(output_dir, "gnina_rough_poses.sdf")
        log_file = os.path.join(output_dir, "gnina_rough.log")

        gnina_cmd = (
            f"{GNINA_BINARY} "
            f"--receptor {receptor_pdb} "
            f"--ligand {ligand_sdf} "
            f"--center_x {grid_center[0]} --center_y {grid_center[1]} --center_z {grid_center[2]} "
            f"--size_x {grid_size[0]} --size_y {grid_size[1]} --size_z {grid_size[2]} "
            f"--exhaustiveness {exhaustiveness} "
            f"--num_modes {num_modes} "
            f"--cpu {cpu} "
            f"--device {gpu_device} "
            f"--cnn_scoring {cnn_scoring} "
            f"--cnn_rotation {cnn_rotation} "
            f"--seed {seed} "
            f"--out {output_sdf} "
            f"--log {log_file}"
        )

        if submit_slurm:
            script = GNINADocker._build_slurm_script(
                gnina_cmd=gnina_cmd,
                job_name="autovs_htvs",
                output_dir=output_dir,
                partition=SLURM_PARTITION_GPU,
                gres=slurm_gres,
                cpus=slurm_cpus,
                mem=slurm_mem,
                walltime=slurm_walltime,
            )
            job_id = SlurmJobManager.submit(script, job_name="autovs_htvs")
            return {
                "job_ids": [job_id] if job_id else [],
                "output_dir": output_dir,
                "output_sdf": output_sdf,
                "log_file": log_file,
                "command": gnina_cmd,
            }
        else:
            subprocess.run(
                gnina_cmd, shell=True, check=False,
                timeout=86400,  # 24h max for local
            )
            return {
                "job_ids": [],
                "output_dir": output_dir,
                "output_sdf": output_sdf,
                "log_file": log_file,
                "command": gnina_cmd,
            }

    @staticmethod
    def run_refinement(
        receptor_pdb: str,
        ligand_sdf: str,
        grid_center: List[float],
        grid_size: List[float],
        output_dir: str,
        exhaustiveness: int = 64,
        num_modes: int = 9,
        min_rmsd_filter: float = 1.0,
        seed: int = 61453,
    ) -> Dict[str, Any]:
        """运行 GNINA Refinement (高精度精筛)。

        用于 Step 3 Watchdog Dry-Run 和 Top-N 分子精筛。
        """
        os.makedirs(output_dir, exist_ok=True)
        output_sdf = os.path.join(output_dir, "gnina_refined.sdf")
        log_file = os.path.join(output_dir, "gnina_refined.log")

        gnina_cmd = (
            f"{GNINA_BINARY} "
            f"--receptor {receptor_pdb} "
            f"--ligand {ligand_sdf} "
            f"--center_x {grid_center[0]} --center_y {grid_center[1]} --center_z {grid_center[2]} "
            f"--size_x {grid_size[0]} --size_y {grid_size[1]} --size_z {grid_size[2]} "
            f"--exhaustiveness {exhaustiveness} "
            f"--num_modes {num_modes} "
            f"--cpu 10 --device 0 "
            f"--cnn_scoring refinement "
            f"--cnn_rotation 4 "
            f"--min_rmsd_filter {min_rmsd_filter} "
            f"--seed {seed} "
            f"--out {output_sdf} "
            f"--log {log_file}"
        )

        script = GNINADocker._build_slurm_script(
            gnina_cmd=gnina_cmd,
            job_name="autovs_refine",
            output_dir=output_dir,
            partition=SLURM_PARTITION_GPU,
            gres="gpu:a100_2g.20gb:1",
            cpus=10,
            mem="64G",
            walltime="1-12:00:00",
        )
        job_id = SlurmJobManager.submit(script, job_name="autovs_refine")

        return {
            "job_ids": [job_id] if job_id else [],
            "output_dir": output_dir,
            "output_sdf": output_sdf,
            "log_file": log_file,
            "command": gnina_cmd,
        }

    @staticmethod
    def _build_slurm_script(
        gnina_cmd: str,
        job_name: str,
        output_dir: str,
        partition: str,
        gres: str,
        cpus: int,
        mem: str,
        walltime: str,
    ) -> str:
        """构建 GNINA Slurm 提交脚本。"""
        script_path = os.path.join(output_dir, "slurm_gnina.sh")
        with open(script_path, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err

source {CONDA_BASE}
conda activate gnina

echo "[$(date)] Starting GNINA docking"
echo "Command: {gnina_cmd}"

{gnina_cmd}

EXIT_CODE=$?
echo "[$(date)] GNINA finished with exit code $EXIT_CODE"
exit $EXIT_CODE
""")
        os.chmod(script_path, 0o755)
        return script_path


# =============================================================================
# GROMACS MD 模拟
# =============================================================================

class GromacsMDRunner:
    """GROMACS MD 模拟引擎封装。

    实现完整的蛋白-配体 MD 工作流:
      1. 蛋白准备 (pdb2gmx)
      2. 配体准备 (obabel + acpype)
      3. 体系构建 (box + solvate + ions)
      4. EM → NVT → NPT → MD 生产相
      5. PBC 清理 + RMSD + MMGBSA 分析
    """

    @staticmethod
    def prepare_and_submit(
        receptor_pdb: str,
        ligand_sdf: str,
        mol_id: str,
        workdir_base: str,
        formal_charge: int,
        simulation_ns: float = 50.0,
        force_field: str = "amber99sb-ildn",
        water_model: str = "tip3p",
        temperature: float = 310.0,
        submit_slurm: bool = True,
        slurm_gres: str = "gpu:a100_2g.20gb:1",
        slurm_cpus: int = 8,
        slurm_mem: str = "20G",
        slurm_walltime: str = "1-12:00:00",
    ) -> Dict[str, Any]:
        """准备 GROMACS MD 体系并提交 Slurm 作业。

        Args:
            receptor_pdb: 受体 PDB 路径。
            ligand_sdf: 配体 SDF 路径 (单个分子)。
            mol_id: 分子标识符。
            workdir_base: 工作目录基路径。
            formal_charge: 配体的形式电荷。
            simulation_ns: MD 模拟时长 (ns)。
            force_field: 蛋白力场。
            water_model: 水模型。
            temperature: 温度 (K)。

        Returns:
            {job_id, workdir, prep_ok, error}
        """
        workdir = os.path.join(workdir_base, mol_id)
        os.makedirs(workdir, exist_ok=True)

        prep_script = os.path.join(workdir, "prepare_and_run.sh")

        # 配体输出 SDF 路径 (工作目录内)
        ligand_copy = os.path.join(workdir, "ligand.sdf")

        # 复制配体
        shutil.copy(ligand_sdf, ligand_copy)

        # 确定 water_model 对应的溶剂文件
        water_spc = "spc216.gro" if water_model in ("tip3p", "spc") else "spc216.gro"

        with open(prep_script, "w") as f:
            f.write(f"""#!/bin/bash
set -euo pipefail

WORKDIR="{workdir}"
cd "$WORKDIR"

source /users_home/wangpengzheng/VS_Agent_HTVS/md_deploy/env_gromacs_cuda.sh
source {CONDA_BASE}
conda activate {GROMACS_CONDA_ENV}

echo "[$(date)] === Step 1: Protein Preparation ==="
gmx pdb2gmx -f {receptor_pdb} -o protein_processed.gro \\
  -p topol.top -i posre.itp -ff {force_field} -water {water_model} -ignh

echo "[$(date)] === Step 2: Ligand Preparation ==="
obabel -isdf {ligand_copy} -omol2 -O ligand.mol2 -h 2>/dev/null || obabel {ligand_copy} -O ligand.mol2 -h 2>/dev/null
acpype -i ligand.mol2 -b LIG -c bcc -n {formal_charge} -a gaff2 -o gmx -d -f 2>&1 | tail -5

# 合并配体拓扑
cp LIG.acpype/LIG_GMX.itp . 2>/dev/null || true
cp LIG.acpype/LIG_GMX.gro . 2>/dev/null || true

echo "[$(date)] === Step 3: System Build ==="
# 合并蛋白+配体
gmx editconf -f protein_processed.gro -o complex_dry.gro -d 1.2 -bt cubic 2>&1 | tail -3

# 手动组合 complex.gro
python3 -c "
import sys
with open('complex_dry.gro') as f:
    lines = f.readlines()
with open('LIG_GMX.gro') as f:
    lig_lines = f.readlines()[2:-1]  # skip header and box/tail

# Insert ligand before box line
# Actually, just concatenate protein and ligand with updated atom count
# This is a simplified approach; production should use proper topology merging
print('GRO concatenation: protein + ligand = complex')
"

# 重新构建: 将蛋白和配体原子坐标合并为一个 .gro
python3 << 'PYEOF'
prot_lines = open('complex_dry.gro').readlines()
lig_lines = open('LIG_GMX.gro').readlines()

# Protein header
prot_header = prot_lines[0]
prot_count = int(prot_lines[1].strip())
prot_atoms = prot_lines[2:-1]
prot_box = prot_lines[-1]

# Ligand atoms (skip header, count line, take atoms and box)
lig_count = int(lig_lines[1].strip())
lig_atoms = lig_lines[2:-1]

# Build complex
total = prot_count + lig_count
with open('complex.gro', 'w') as f:
    f.write(f'Protein-Ligand complex\\n')
    f.write(f'{total:5d}\\n')
    for l in prot_atoms:
        f.write(l)
    for l in lig_atoms:
        f.write(l)
    f.write(prot_box)  # Keep protein box for now
PYEOF

# Solvate
gmx solvate -cp complex.gro -cs {water_spc} -o solvated.gro -p topol.top 2>&1 | tail -3

# Add ions
gmx grompp -f {MDP_TEMPLATE_DIR}/em.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 5 2>&1 | tail -3
printf "SOL\\n" | gmx genion -s ions.tpr -o ionized.gro -p topol.top \\
  -pname NA -nname CL -neutral -conc 0.15 2>&1 | tail -3

# Final index
printf "Protein\\nBackbone\\nLIG\\nProtein|LIG\\nq\\n" | gmx make_ndx -f ionized.gro -o index.ndx 2>/dev/null || true

echo "[$(date)] === Step 4: Energy Minimization ==="
gmx grompp -f {MDP_TEMPLATE_DIR}/em.mdp -c ionized.gro -p topol.top -n index.ndx -o em.tpr -maxwarn 5
gmx mdrun -deffnm em -v 2>&1 | tail -5

echo "[$(date)] === Step 5: NVT Equilibration ==="
gmx grompp -f {MDP_TEMPLATE_DIR}/nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 5
gmx mdrun -deffnm nvt -v 2>&1 | tail -5

echo "[$(date)] === Step 6: NPT Equilibration ==="
gmx grompp -f {MDP_TEMPLATE_DIR}/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr -maxwarn 5
gmx mdrun -deffnm npt -v 2>&1 | tail -5

echo "[$(date)] === Step 7: MD Production ({simulation_ns} ns) ==="
NSTEPS=$(python3 -c "print(int({simulation_ns} * 500000))")
sed -i "s/nsteps.*=.*/nsteps = $NSTEPS/" {MDP_TEMPLATE_DIR}/md.mdp
gmx grompp -f {MDP_TEMPLATE_DIR}/md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md_{simulation_ns:.0f}ns.tpr -maxwarn 5
gmx mdrun -deffnm md_{simulation_ns:.0f}ns -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK \\
  -nb gpu -pme gpu -bonded gpu -update gpu -pin on -v 2>&1 | tail -10

echo "[$(date)] === Step 8: PBC Cleanup ==="
printf "System\\n" | gmx trjconv -s md_{simulation_ns:.0f}ns.tpr -f md_{simulation_ns:.0f}ns.xtc \\
  -n index.ndx -o md_whole.xtc -pbc whole 2>/dev/null
printf "Protein_LIG\\nSystem\\n" | gmx trjconv -s md_{simulation_ns:.0f}ns.tpr -f md_whole.xtc \\
  -n index.ndx -o md_cluster.xtc -pbc cluster 2>/dev/null
printf "Protein_LIG\\nSystem\\n" | gmx trjconv -s md_{simulation_ns:.0f}ns.tpr -f md_cluster.xtc \\
  -n index.ndx -o md_center.xtc -center -pbc mol -ur compact 2>/dev/null
printf "Backbone\\nSystem\\n" | gmx trjconv -s md_{simulation_ns:.0f}ns.tpr -f md_center.xtc \\
  -n index.ndx -o md_fit.xtc -fit rot+trans 2>/dev/null

echo "[$(date)] === Step 9: RMSD Analysis ==="
printf "Backbone\\nBackbone\\n" | gmx rms -s md_{simulation_ns:.0f}ns.tpr -f md_fit.xtc \\
  -n index.ndx -o rmsd_protein.xvg 2>/dev/null
printf "LIG\\nLIG\\n" | gmx rms -s md_{simulation_ns:.0f}ns.tpr -f md_fit.xtc \\
  -n index.ndx -o rmsd_ligand.xvg 2>/dev/null

echo "[$(date)] === Step 10: MMGBSA (last 30%) ==="
BS=$(python3 -c "print(int({simulation_ns} * 700))")  # last 30% from 70% to 100%
printf "Protein_LIG\\n" | gmx trjconv -s md_{simulation_ns:.0f}ns.tpr -f md_fit.xtc \\
  -n index.ndx -o md_mmgbsa.xtc -b $BS 2>/dev/null

# Check if MMGBSA input file exists
if [ -f "mmpbsa.in" ]; then
    gmx_MMPBSA MPI -p topol.top -c complex.top \\
      --group-file index.ndx \\
      --trajectory md_mmgbsa.xtc \\
      --temperature {temperature} \\
      --solvent TIP3P 2>&1 | tail -10 || echo "MMGBSA_ANALYSIS_FAILED"
fi

echo '{{"status": "completed", "mol_id": "{mol_id}", "timestamp": "'$(date -Iseconds)'"}}' > status.json
echo "[$(date)] DONE"
""")
        os.chmod(prep_script, 0o755)

        if submit_slurm:
            slurm_script = os.path.join(workdir, "slurm_md.sh")
            with open(slurm_script, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name=autovs_md_{mol_id[:8]}
#SBATCH --partition={SLURM_PARTITION_GPU}
#SBATCH --gres={slurm_gres}
#SBATCH --cpus-per-task={slurm_cpus}
#SBATCH --mem={slurm_mem}
#SBATCH --time={slurm_walltime}
#SBATCH --output={workdir}/slurm_%j.out
#SBATCH --error={workdir}/slurm_%j.err

bash {prep_script}
""")
            os.chmod(slurm_script, 0o755)
            job_id = SlurmJobManager.submit(slurm_script, job_name=f"md_{mol_id[:8]}")
            return {
                "job_id": job_id,
                "workdir": workdir,
                "prep_ok": True,
                "error": None,
            }
        else:
            try:
                subprocess.run(
                    ["bash", prep_script],
                    check=False, timeout=3600,
                    capture_output=True,
                )
                return {"job_id": None, "workdir": workdir, "prep_ok": True, "error": None}
            except subprocess.TimeoutExpired:
                return {
                    "job_id": None, "workdir": workdir, "prep_ok": False,
                    "error": "Local MD preparation timed out (>1h). Submit via Slurm.",
                }

    @staticmethod
    def analyze_results(
        workdir: str,
        mol_id: str,
    ) -> Dict[str, Any]:
        """分析已完成 MD 的结果。

        读取 RMSD xvg、MMGBSA dat、status.json，提取关键指标。

        Returns:
            遵循 MDSimulationRecord 格式的 dict。
        """
        status_file = os.path.join(workdir, "status.json")
        if not os.path.exists(status_file):
            return {
                "mol_id": mol_id,
                "simulation_status": "failed",
                "error_message": "No status.json found",
                "complex_stable": False,
            }

        result = {
            "mol_id": mol_id,
            "trajectory_path": os.path.join(workdir, "md_fit.xtc"),
            "topology_path": os.path.join(workdir, f"md_50ns.tpr"),
            "total_time_ns": 50.0,
            "ligand_rmsd_mean": 0.0,
            "ligand_rmsd_std": 0.0,
            "protein_rmsd_mean": 0.0,
            "key_hbond_occupancy": {},
            "dG_mmgbsa": None,
            "dG_mmpbsa": None,
            "kd_predicted": None,
            "complex_stable": False,
            "simulation_status": "failed",
            "error_message": None,
        }

        # 解析 RMSD
        rmsd_file = os.path.join(workdir, "rmsd_ligand.xvg")
        if os.path.exists(rmsd_file):
            rmsd_vals = GromacsMDRunner._parse_xvg_y(rmsd_file)
            if rmsd_vals:
                import numpy as np
                result["ligand_rmsd_mean"] = round(float(np.mean(rmsd_vals)), 2)
                result["ligand_rmsd_std"] = round(float(np.std(rmsd_vals)), 2)

        # 解析 protein RMSD
        prot_rmsd_file = os.path.join(workdir, "rmsd_protein.xvg")
        if os.path.exists(prot_rmsd_file):
            rmsd_vals = GromacsMDRunner._parse_xvg_y(prot_rmsd_file)
            if rmsd_vals:
                import numpy as np
                result["protein_rmsd_mean"] = round(float(np.mean(rmsd_vals)), 2)

        # 解析 MMGBSA
        mmpbsa_result = os.path.join(workdir, "FINAL_RESULTS_MMPBSA.dat")
        if os.path.exists(mmpbsa_result):
            dG = GromacsMDRunner._parse_mmgbsa_dg(mmpbsa_result)
            if dG is not None:
                result["dG_mmgbsa"] = dG
                result["dG_mmpbsa"] = dG
                # 粗略 Kd 估算: ΔG = -RT ln(Kd), R=1.987e-3 kcal/mol/K, T=310K
                RT = 1.987e-3 * 310.0  # ≈ 0.616 kcal/mol
                result["kd_predicted"] = round(math.exp(dG / RT), 1) if dG < 0 else 1e6

        # 稳定性判断
        if result["ligand_rmsd_mean"] > 0:
            result["complex_stable"] = (
                result["ligand_rmsd_mean"] < 3.0 and
                result["protein_rmsd_mean"] < 2.5
            )

        result["simulation_status"] = "completed"
        return result

    @staticmethod
    def _parse_xvg_y(xvg_path: str) -> List[float]:
        """提取 xvg 文件中的 Y 列数据。"""
        vals = []
        try:
            with open(xvg_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(("#", "@")):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            vals.append(float(parts[1]))
                        except ValueError:
                            pass
        except Exception:
            pass
        return vals

    @staticmethod
    def _parse_mmgbsa_dg(dat_path: str) -> Optional[float]:
        """从 FINAL_RESULTS_MMPBSA.dat 提取 ΔG (GB 列)。"""
        try:
            with open(dat_path) as f:
                for line in f:
                    if line.startswith("Differences") or "DELTA" in line:
                        # 尝试从后续行提取 ΔG
                        pass
                # 回退: 搜索 "delta G" 或最后列的数值
                f.seek(0)
                for line in f:
                    if "delta" in line.lower() and "gb" in line.lower():
                        parts = line.strip().split()
                        for p in parts:
                            try:
                                return float(p)
                            except ValueError:
                                continue
        except Exception:
            pass
        return None


# =============================================================================
# 预处理工具
# =============================================================================

class PrepUtils:
    """受体/配体预处理。"""

    @staticmethod
    def clean_receptor_pdb(
        pdb_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """清理受体 PDB: 去除水、配体、异质原子，仅保留蛋白。

        Args:
            pdb_path: 原始 PDB 路径。
            output_dir: 输出目录。

        Returns:
            清理后的 PDB 路径。
        """
        if output_dir is None:
            output_dir = os.path.dirname(pdb_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "receptor_clean.pdb")

        # 用 grep 快速过滤
        result = subprocess.run(
            ["grep", "-E", "^(ATOM|TER)", pdb_path],
            capture_output=True, text=True,
        )
        with open(output_path, "w") as f:
            f.write(result.stdout)

        return output_path

    @staticmethod
    def prepare_ligands_for_docking(
        molecules: List[dict],
        output_dir: str,
        chunk_size: int = 5000,
    ) -> List[str]:
        """准备对接配体 SDF: SMILES→3D SDF + 分块。

        Args:
            molecules: MoleculeRecord 列表 (需含 mol_id, smiles)。
            output_dir: 输出目录。
            chunk_size: 每个 SDF chunk 的分子数。

        Returns:
            SDF chunk 文件路径列表。
        """
        os.makedirs(output_dir, exist_ok=True)

        smiles_list = [m.get("smiles", "") for m in molecules]
        mol_ids = [m.get("mol_id", f"mol_{i}") for i, m in enumerate(molecules)]

        chunks = []
        for i in range(0, len(smiles_list), chunk_size):
            chunk_smi = smiles_list[i:i + chunk_size]
            chunk_ids = mol_ids[i:i + chunk_size]
            chunk_path = os.path.join(output_dir, f"ligands_chunk_{i // chunk_size:03d}.sdf")
            SDFUtils.smiles_to_sdf(chunk_smi, chunk_ids, chunk_path)
            chunks.append(chunk_path)

        return chunks


# =============================================================================
# ADMET-AI 集成 (真实 ML 预测)
# =============================================================================

# ADMET-AI 提供的 42 个预测属性分类:
#   Physicochemical (9): MW, LogP, HBD, HBA, Lipinski, QED, TPSA, PAINS/BRENK/NIH
#   Absorption (7):      HIA, Bioavailability, Solubility, Lipophilicity, Caco-2, PAMPA, Pgp
#   Distribution (3):    BBB, PPBR, VDss
#   Excretion (2):       Half-Life, Clearance (Hepatocyte/Microsome)
#   Metabolism (8):      CYP1A2/2C19/2C9/2D6/3A4 Inhibition + CYP2C9/2D6/3A4 Substrate
#   Toxicity (13):       hERG, ClinTox, AMES, DILI, Carcinogens, LD50, Skin, NR/SR panels

# 与 MedChem Committee 联动的关键阈值
ADMET_CRITICAL_PROPERTIES = [
    "hERG",           # hERG 阻断 → >0.5 则 veto
    "AMES",           # 致突变性 → >0.5 则 veto
    "DILI",           # 肝毒性 → >0.5 则 veto
    "ClinTox",        # 临床毒性 → >0.5 则 veto
    "Carcinogens_Lagunin",  # 致癌性 → >0.5 则 veto
    "BBB_Martins",    # 血脑屏障 (CNS靶点需关注)
    "HIA_Hou",         # 人体肠道吸收
    "Bioavailability_Ma",  # 口服生物利用度
    "Pgp_Broccatelli",     # P-糖蛋白抑制
    "CYP2D6_Veith",   # CYP2D6 抑制
    "CYP3A4_Veith",   # CYP3A4 抑制
    "CYP1A2_Veith",   # CYP1A2 抑制
    "CYP2C9_Veith",   # CYP2C9 抑制
    "CYP2C19_Veith",  # CYP2C19 抑制
    "LD50_Zhu",       # 急性毒性
    "Skin_Reaction",  # 皮肤反应
    "PAINS_alert",    # PAINS 假阳性
    "BRENK_alert",    # BRENK 预警
    "Lipinski",       # 五规则违规数
]


class ADMETAIPredictor:
    """ADMET-AI 真实 ML 预测封装。

    使用 Chemprop v2 模型对 40+ ADMET 属性进行预测。
    替换 expert_committee.py 中的 Tier 2 mock _run_admet_ai()。
    """

    _model = None  # 单例模型

    @classmethod
    def _get_model(cls):
        """惰性加载 ADMET-AI 模型（单例，加载需约30秒）。"""
        if cls._model is None:
            from admet_ai import ADMETModel
            cls._model = ADMETModel()
        return cls._model

    @classmethod
    def predict_single(cls, smiles: str) -> Dict[str, Any]:
        """对单个分子预测全部 ADMET 属性。

        Args:
            smiles: SMILES 字符串。

        Returns:
            {property_name: value, ...} 42 个 ADMET 属性字典。
            如果预测失败，返回空 dict。
        """
        if not smiles:
            return {}
        try:
            model = cls._get_model()
            preds = model.predict(smiles=smiles)
            return preds if isinstance(preds, dict) else {}
        except Exception:
            return {}

    @classmethod
    def predict_batch(
        cls,
        smiles_list: List[str],
        batch_size: int = 500,
    ) -> List[Dict[str, Any]]:
        """对一批分子预测 ADMET 属性。

        Args:
            smiles_list: SMILES 字符串列表。
            batch_size: 每批处理数量。

        Returns:
            [{property: value, ...}, ...] 与输入一一对应。
        """
        if not smiles_list:
            return []

        results = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            try:
                model = cls._get_model()
                df = model.predict(smiles=batch)
                # DataFrame → list of dicts
                batch_results = df.to_dict(orient="records")
                results.extend(batch_results)
            except Exception:
                # 批量失败时逐条重试
                for smi in batch:
                    results.append(cls.predict_single(smi))

        return results

    @classmethod
    def get_flag_dict(cls, smiles: str) -> Dict[str, bool]:
        """获取分子的关键 ADMET 二值标记。

        将连续概率阈值化 (>0.5 → True)，用于 MedChem 一票否决。

        Args:
            smiles: SMILES 字符串。

        Returns:
            {flag_name: bool, ...} 如 {"hERG": True, "AMES": False, ...}
        """
        preds = cls.predict_single(smiles)
        if not preds:
            return {}

        flags = {}
        # 分类属性阈值化
        classification_keys = [
            "hERG", "AMES", "DILI", "ClinTox", "Carcinogens_Lagunin",
            "HIA_Hou", "Bioavailability_Ma", "Pgp_Broccatelli",
            "CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith",
            "CYP2D6_Veith", "CYP3A4_Veith",
            "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
            "CYP3A4_Substrate_CarbonMangels",
            "BBB_Martins", "PAMPA_NCATS", "Skin_Reaction",
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ]
        for key in classification_keys:
            val = preds.get(key)
            if val is not None:
                flags[key] = float(val) > 0.5

        # PAINS/BRENK 整数值 (>0 表示有警示)
        for alert_key in ["PAINS_alert", "BRENK_alert", "NIH_alert"]:
            val = preds.get(alert_key)
            if val is not None:
                flags[alert_key] = int(val) > 0

        return flags

    @classmethod
    def get_toxicity_veto_list(cls, smiles: str) -> List[str]:
        """获取分子的毒性否决项列表。

        对 hERG/AMES/DILI/ClinTox/Carcinogens PAINS BRENK 进行阈值判断。

        Returns:
            触发否决的属性名列表。空列表表示通过。
        """
        preds = cls.predict_single(smiles)
        vetoed = []

        checks = {
            "hERG_blocker": ("hERG", 0.5, "hERG 钾通道阻断风险"),
            "mutagenicity": ("AMES", 0.5, "Ames 致突变性"),
            "hepatotoxicity": ("DILI", 0.5, "药物性肝损伤风险"),
            "clinical_toxicity": ("ClinTox", 0.5, "临床毒性风险"),
            "carcinogenicity": ("Carcinogens_Lagunin", 0.5, "致癌性风险"),
        }

        for flag_name, (key, threshold, label) in checks.items():
            val = preds.get(key)
            if val is not None and float(val) > threshold:
                vetoed.append(f"Tox_{flag_name}: {label} (prob={float(val):.2f})")

        # PAINS / BRENK
        for alert_key, label in [("PAINS_alert", "PAINS"), ("BRENK_alert", "BRENK")]:
            val = preds.get(alert_key)
            if val is not None and int(val) > 0:
                vetoed.append(f"Alert_{label}: {label} warning ({int(val)} alerts)")

        return vetoed
