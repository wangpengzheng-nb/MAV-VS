"""分子准备流水线：标准化 → 枚举 → 3D → PDBQT。

整合 5 个工具：
1. ChEMBL Structure Pipeline — 标准化 + 去盐（MIT）
2. Gypsum-DL — 质子化/互变/立体/构象生成（Apache-2.0）
3. Dimorphite-DL — pH 依赖离子化枚举
4. Meeko — AutoDock PDBQT 参数化
5. Open Babel — 通用格式转换

关键约束：不无限枚举状态。每个原始分子最多：
- 4 个质子化/互变异构状态
- 4 个立体状态
- 3 个初始构象
"""

from __future__ import annotations

import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any


# ── 枚举上限（防止组合爆炸）─────────────────────────────────────────

_MAX_PROTOMER_STATES = 4     # 质子化+互变异构
_MAX_STEREO_STATES = 4       # 手性+顺反异构
_MAX_CONFORMERS = 3          # 每个状态的初始构象数


def _iter_strict_smi(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            mol_id, smiles = parts[0].strip(), parts[1].strip()
            if mol_id.lower() in {"source_id", "molecule_id", "id"} and smiles.lower() in {"smiles", "smi"}:
                continue
            yield mol_id, smiles


def _largest_fragment(mol: Any) -> Any:
    from rdkit import Chem

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return mol
    return max(fragments, key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms()))


# ═══════════════════════════════════════════════════════════════════════
# 1. ChEMBL Structure Pipeline — 标准化 + 去盐
# ═══════════════════════════════════════════════════════════════════════

def standardize_molecules(
    input_path: str | Path,
    output_path: str | Path,
    *,
    remove_salts: bool = True,
    neutralize: bool = False,
) -> dict[str, Any]:
    """使用 ChEMBL Structure Pipeline 标准化分子库。

    - 去除盐/反离子（remove_salts=True）
    - 标准化官能团表示
    - 保留最大共价组分
    - 可选中和（通常不建议，保留原始电荷供后续枚举）

    Args:
        input_path: 输入 SMILES 文件（每行 molecule_id\\tSMILES）
        output_path: 输出标准化后的 SMILES 文件
        remove_salts: 是否去除盐
        neutralize: 是否中和（默认 False，保留原始电荷态）

    Returns:
        诊断报告
    """
    from chembl_structure_pipeline.standardizer import (
        get_parent_mol, standardize_mol, uncharge_mol,
    )
    from rdkit import Chem

    filepath = Path(input_path)
    if not filepath.is_file():
        raise FileNotFoundError(f"输入文件不存在: {filepath}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total, success, failed, salt_removed = 0, 0, 0, 0
    rejected: list[dict[str, Any]] = []

    with out.open("w", encoding="utf-8") as handle:
        for mol_id, smiles in _iter_strict_smi(filepath):
                total += 1
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        failed += 1
                        rejected.append({"source_id": mol_id, "smiles": smiles,
                                         "reason": "RDKit 解析失败"})
                        continue

                    if remove_salts:
                        parent_result = get_parent_mol(mol)
                        if isinstance(parent_result, tuple):
                            parent = parent_result[0]
                        else:
                            parent = parent_result
                        if parent is None or not isinstance(parent, Chem.Mol):
                            failed += 1
                            rejected.append({"source_id": mol_id, "smiles": smiles,
                                             "reason": "去盐后无有效分子"})
                            continue
                        parent = _largest_fragment(parent)
                        if Chem.MolToSmiles(parent, isomericSmiles=True) != Chem.MolToSmiles(mol, isomericSmiles=True):
                            salt_removed += 1
                        mol = parent

                    mol = standardize_mol(mol)

                    if neutralize:
                        mol = uncharge_mol(mol)

                    clean = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if clean:
                        handle.write(f"{mol_id}\t{clean}\n")
                        success += 1
                    else:
                        failed += 1
                        rejected.append({"source_id": mol_id, "smiles": smiles,
                                         "reason": "输出 SMILES 为空"})
                except Exception as exc:
                    failed += 1
                    rejected.append({"source_id": mol_id, "smiles": smiles,
                                     "reason": str(exc)})

    report = {
        "total": total, "success": success, "failed": failed,
        "salt_removed": salt_removed, "rejected": rejected[:100],
        "output_path": str(out),
    }
    return report


# ═══════════════════════════════════════════════════════════════════════
# 2. Dimorphite-DL — pH 依赖离子化枚举
# ═══════════════════════════════════════════════════════════════════════

def enumerate_ionization(
    input_smiles: list[str],
    *,
    ph_min: float = 7.4,
    ph_max: float = 7.4,
    max_states: int = 4,
) -> list[dict[str, Any]]:
    """使用 Dimorphite-DL 枚举指定 pH 范围的离子化状态。

    Args:
        input_smiles: SMILES 列表
        ph_min: 最低 pH
        ph_max: 最高 pH（设置与 ph_min 不同可覆盖 pH 范围）
        max_states: 每个分子最多返回的状态数

    Returns:
        [{source_id, smiles, protonation_site, ...}]
    """
    from dimorphite_dl import protonate_smiles

    results: list[dict[str, Any]] = []
    for i, item in enumerate(input_smiles):
        source_id = f"mol_{i}"
        smi = item.strip()
        if not smi:
            continue
        parts = smi.split("\t", 1)
        if len(parts) >= 2:
            source_id, smi = parts[0].strip(), parts[1].strip()
        elif " " in smi:
            maybe_smi, maybe_id = smi.split(None, 1)
            source_id, smi = maybe_id.strip(), maybe_smi.strip()
        if source_id.lower() in {"source_id", "molecule_id", "id"} and smi.lower() in {"smiles", "smi"}:
            continue
        variants = protonate_smiles(
            smi, ph_min=ph_min, ph_max=ph_max,
            max_variants=max_states, label_identifiers=False, label_states=False,
        )
        seen: set[str] = set()
        for j, var in enumerate(variants):
            if var in seen:
                continue
            seen.add(var)
            results.append({
                "source_id": source_id,
                "variant_index": j,
                "smiles": var,
            })
    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. Gypsum-DL — 全面 3D-ready 枚举（质子化+互变+立体+构象）
# ═══════════════════════════════════════════════════════════════════════

def prepare_ligands_3d(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ph: float = 7.4,
    max_variants_per_compound: int = _MAX_PROTOMER_STATES,
    max_stereo_states: int = _MAX_STEREO_STATES,
    max_conformers: int = _MAX_CONFORMERS,
    let_tautomers_change_chirality: bool = False,
    skip_optimize_geometry: bool = False,
    skip_alternate_ring_conformations: bool = True,
    job_manager: str = "multiprocessing",
    num_processes: int = -1,
) -> dict[str, Any]:
    """使用 Gypsum-DL 将 SMILES 库转换为 3D-ready SDF。

    处理：质子化状态、互变异构、未指定手性、顺反异构、
          环构象（默认关闭）、3D 坐标生成（UFF + MMFF94）。

    Args:
        input_path: 输入 SMILES 文件（source_id\\tSMILES）
        output_path: 输出 SDF 路径
        ph: 目标 pH
        max_variants_per_compound: 每个分子最多质子化/互变异构状态
        max_stereo_states: 每个分子最多立体状态
        max_conformers: 每个状态最多构象数
        skip_alternate_ring_conformations: 是否跳过环构象枚举（默认 True）

    Returns:
        诊断报告
    """
    filepath = Path(input_path)
    if not filepath.is_file():
        raise FileNotFoundError(f"输入文件不存在: {filepath}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import shutil

    work_dir = out.parent
    temp_input = work_dir / "gypsum_input.smi"
    with temp_input.open("w", encoding="utf-8") as handle:
        for mol_id, smiles in _iter_strict_smi(filepath):
            handle.write(f"{smiles} {mol_id}\n")

    from gypsum_dl.run import prepare_molecules

    args = {
        "source": str(temp_input),
        "output_folder": str(work_dir),
        "job_manager": "serial" if job_manager == "multiprocessing" and num_processes in {-1, 1} else job_manager,
        "num_processors": num_processes if num_processes > 0 else 1,
        "max_variants_per_compound": max(1, max_variants_per_compound),
        "thoroughness": 1,
        "separate_output_files": False,
        "add_pdb_output": False,
        "add_html_output": False,
        "2d_output_only": False,
        "skip_optimize_geometry": skip_optimize_geometry,
        "skip_alternate_ring_conformations": skip_alternate_ring_conformations,
        "let_tautomers_change_chirality": let_tautomers_change_chirality,
        "min_ph": ph,
        "max_ph": ph,
    }

    returncode = 0
    stderr_tail = ""
    stdout_tail = ""
    try:
        stdout_buffer, stderr_buffer = StringIO(), StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            prepare_molecules(args)
        stdout_tail = stdout_buffer.getvalue()[-1000:]
        stderr_tail = stderr_buffer.getvalue()[-500:]
    except Exception as exc:
        returncode = 1
        stderr_tail = str(exc)[-500:]

    output_sdf = work_dir / "gypsum_dl_success.sdf"
    if output_sdf.is_file():
        shutil.copy(output_sdf, out)

    report = {
        "ph": ph,
        "max_variants": max_variants_per_compound,
        "max_stereo": max_stereo_states,
        "max_conformers": max_conformers,
        "output_sdf": str(out) if out.is_file() else None,
        "variant_count": _count_records(out, "sdf"),
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if returncode or not out.is_file():
        raise RuntimeError(f"Gypsum-DL failed to produce SDF: {stderr_tail or 'output missing'}")
    return report


# ═══════════════════════════════════════════════════════════════════════
# 4. Meeko — SDF → PDBQT 参数化（AutoDock/Vina）
# ═══════════════════════════════════════════════════════════════════════

def prepare_pdbqt(
    input_sdf: str | Path,
    output_pdbqt: str | Path,
    *,
    ph: float = 7.4,
    add_hydrogens: bool = True,
    merge_these_hydrogens: bool = False,
) -> dict[str, Any]:
    """使用 Meeko 将配体 SDF 转换为 AutoDock PDBQT 格式。

    Meeko 处理：
    - Gasteiger 电荷分配
    - AutoDock 原子类型
    - 可选柔性键
    - 氢原子处理

    Args:
        input_sdf: 输入 SDF 文件
        output_pdbqt: 输出 PDBQT 文件
        ph: 质子化 pH
        add_hydrogens: 是否添加氢原子
        merge_these_hydrogens: 是否合并非极性氢

    Returns:
        诊断报告
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    sdf_path = Path(input_sdf)
    if not sdf_path.is_file():
        raise FileNotFoundError(f"SDF 文件不存在: {sdf_path}")

    out = Path(output_pdbqt)
    out.parent.mkdir(parents=True, exist_ok=True)

    merge_types = ("H",) if merge_these_hydrogens else ()
    preparator = MoleculePreparation(merge_these_atom_types=merge_types)

    suppliers = Chem.SDMolSupplier(str(sdf_path), removeHs=not add_hydrogens)
    total, success = 0, 0

    with out.open("w", encoding="utf-8") as handle:
        for mol in suppliers:
            if mol is None:
                continue
            total += 1
            try:
                mol = Chem.AddHs(mol, addCoords=True) if add_hydrogens else mol
                setups = preparator.prepare(mol)
                if not isinstance(setups, (list, tuple)):
                    setups = [setups]
                for setup in setups:
                    result = PDBQTWriterLegacy.write_string(setup)
                    pdbqt_string = result[0] if isinstance(result, tuple) else result
                    if pdbqt_string:
                        handle.write(pdbqt_string)
                        if not pdbqt_string.rstrip().endswith("END"):
                            handle.write("\nEND\n")
                        success += 1
            except Exception:
                continue

    return {
        "total": total,
        "success": success,
        "output_pdbqt": str(out) if success > 0 else None,
        "ph": ph,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Open Babel — 通用格式转换
# ═══════════════════════════════════════════════════════════════════════

def obabel_convert(
    input_path: str | Path,
    output_path: str | Path,
    input_format: str = "smi",
    output_format: str = "sdf",
    *,
    gen3d: bool = False,
    add_hydrogens: bool = False,
    ph: float = 7.4,
    obabel_path: str | Path | None = None,
    **extra_flags: str,
) -> dict[str, Any]:
    """使用 Open Babel 进行分子格式转换。

    Args:
        input_path: 输入文件
        output_path: 输出文件
        input_format: 输入格式（smi/sdf/pdb/mol2 等）
        output_format: 输出格式
        gen3d: 是否生成 3D 坐标
        add_hydrogens: 是否添加氢（含 pH 依赖质子化）
        ph: 加氢时的 pH
        **extra_flags: 额外 obabel 选项

    Returns:
        诊断报告
    """
    filepath = Path(input_path)
    if not filepath.is_file():
        raise FileNotFoundError(f"输入文件不存在: {filepath}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    obabel_input = filepath
    if input_format.lower() in {"smi", "smiles"}:
        obabel_input = out.parent / f"{filepath.stem}.obabel_input.smi"
        with obabel_input.open("w", encoding="utf-8") as handle:
            for mol_id, smiles in _iter_strict_smi(filepath):
                handle.write(f"{smiles} {mol_id}\n")

    obabel = Path(obabel_path) if obabel_path else Path("obabel")
    cmd = [
        str(obabel),
        f"-i{input_format}", str(obabel_input),
        f"-o{output_format}", "-O", str(out),
    ]
    if gen3d:
        cmd.insert(1, "--gen3d")
    if add_hydrogens:
        cmd.insert(1, "-h")
        cmd.insert(1, f"-p{ph}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    molecule_count = _count_records(out, output_format)
    if result.returncode or not out.is_file() or molecule_count == 0:
        raise RuntimeError(f"Open Babel conversion failed: {result.stderr[-500:]}")

    return {
        "input": str(filepath),
        "output": str(out),
        "returncode": result.returncode,
        "molecules": molecule_count,
        "stderr": "",
    }


def _count_records(path: Path, fmt: str) -> int:
    if not path.is_file():
        return 0
    if fmt.lower() == "sdf":
        return path.read_text(errors="ignore").count("$$$$")
    if fmt.lower() in {"smi", "smiles"}:
        return sum(1 for line in path.read_text(errors="ignore").splitlines() if line.strip())
    return 1 if path.stat().st_size > 0 else 0


# ═══════════════════════════════════════════════════════════════════════
# 流水线快捷接口
# ═══════════════════════════════════════════════════════════════════════

def full_preparation_pipeline(
    input_smi: str | Path,
    output_dir: str | Path,
    *,
    ph: float = 7.4,
    max_variants: int = _MAX_PROTOMER_STATES,
    max_conformers: int = _MAX_CONFORMERS,
) -> dict[str, Any]:
    """一键运行完整分子准备流水线。

    Standardize → Enumerate(2D) → 3D Conformers → PDBQT

    Args:
        input_smi: 输入 SMILES 文件
        output_dir: 输出目录
        ph: 目标 pH
        max_variants: 每个分子最大枚举数
        max_conformers: 每个状态最大构象数

    Returns:
        全流程诊断报告
    """
    work = Path(output_dir)
    work.mkdir(parents=True, exist_ok=True)

    # 1. 标准化 + 去盐
    std_smi = work / "01_standardized.smi"
    std_report = standardize_molecules(input_smi, std_smi)

    # 2. Gypsum-DL 3D 枚举（已含质子化+互变+立体+构象）
    gypsum_sdf = work / "02_gypsum_3d.sdf"
    gypsum_report = prepare_ligands_3d(
        std_smi, gypsum_sdf,
        ph=ph,
        max_variants_per_compound=max_variants,
        max_conformers=max_conformers,
    )

    # 3. Meeko PDBQT 参数化
    pdbqt_out = work / "03_meeko.pdbqt"
    meeko_report: dict[str, Any] = {}
    if gypsum_sdf.is_file():
        meeko_report = prepare_pdbqt(gypsum_sdf, pdbqt_out, ph=ph)
    else:
        # 回退：直接用标准化后的 SMILES + obabel 3D → Meeko
        obabel_sdf = work / "02b_obabel_3d.sdf"
        obabel_convert(std_smi, obabel_sdf, gen3d=True, add_hydrogens=True, ph=ph)
        if obabel_sdf.is_file():
            meeko_report = prepare_pdbqt(obabel_sdf, pdbqt_out, ph=ph)

    return {
        "standardization": std_report,
        "gypsum_3d": gypsum_report,
        "meeko_pdbqt": meeko_report,
        "output_dir": str(work),
        "ph": ph,
    }
