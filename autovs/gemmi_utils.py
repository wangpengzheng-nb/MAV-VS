"""Gemmi 结构解析工具模块。

提供基于 Gemmi 的 PDB/mmCIF 读取、验证、配体检测、链提取、
口袋残基搜索等功能。所有操作是进程内纯 Python 调用。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Gemmi 是编译型 C++ 绑定，导入开销可忽略
import gemmi


# ── 结构读取与验证 ───────────────────────────────────────────────────

def read_structure(path: str | Path) -> gemmi.Structure:
    """读取 PDB 或 mmCIF 文件，自动检测格式。

    Raises:
        ValueError: 文件格式不支持或结构为空
        FileNotFoundError: 文件不存在
    """
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"结构文件不存在: {filepath}")
    suffix = filepath.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        st = gemmi.read_pdb(str(filepath))
    elif suffix in {".cif", ".mmcif"}:
        doc = gemmi.cif.read(str(filepath))
        st = gemmi.make_structure_from_block(doc.sole_block())
    elif suffix == ".gz":
        # 尝试解压后读取
        if ".pdb" in filepath.stem.lower():
            st = gemmi.read_pdb(str(filepath))
        else:
            raise ValueError(f"无法识别的压缩结构文件: {filepath}")
    else:
        raise ValueError(f"不支持的结构文件格式: {suffix}（需要 .pdb / .cif / .mmcif）")
    atom_count = sum(len(res) for model in st for chain in model for res in chain)
    if atom_count == 0:
        raise ValueError(f"结构文件无原子: {filepath}")
    return st


def validate_structure(path: str | Path) -> dict[str, Any]:
    """验证结构文件质量，返回诊断报告。

    Returns:
        dict 包含:
        - path: 输入文件路径
        - format: PDB 或 mmCIF
        - models: 模型数量
        - chains: 链列表（chain_name, residue_count）
        - atom_count: 总原子数
        - has_ligands: 是否有非蛋白配体
        - ligands: 配体列表（HETATM 且非水/非离子）
        - issues: 潜在问题列表（空字符串表示无问题）
        - resolution: 分辨率（从 REMARK 中提取，mmCIF 更可靠）
    """
    filepath = Path(path)
    report: dict[str, Any] = {
        "path": str(filepath),
        "format": "",
        "models": 0,
        "chains": [],
        "atom_count": 0,
        "has_ligands": False,
        "ligands": [],
        "issues": [],
        "resolution": None,
    }

    try:
        if filepath.suffix.lower() in {".pdb", ".ent", ".gz"}:
            st = gemmi.read_pdb(str(filepath))
            report["format"] = "PDB"
        else:
            st = read_structure(path)
            report["format"] = "mmCIF"
    except Exception as exc:
        report["issues"].append(f"无法读取结构: {exc}")
        return report

    report["models"] = len(st)
    if report["models"] > 1:
        report["issues"].append(f"NMR 结构包含 {len(st)} 个模型；仅处理第一个模型")

    model = st[0]
    chain_info: list[dict[str, Any]] = []
    atom_count = 0
    standard_residues = _STANDARD_AMINO_ACIDS | _NUCLEIC_ACIDS
    skip_het = _SOLVENT_AND_IONS

    for chain in model:
        residues = list(chain)
        chain_info.append({
            "name": chain.name,
            "residue_count": len(residues),
        })
        for residue in residues:
            atom_count += len(residue)
    report["chains"] = chain_info
    report["atom_count"] = atom_count

    # 配体检测：非标准残基且非水/离子
    ligands = find_ligands(st)
    report["has_ligands"] = len(ligands) > 0
    report["ligands"] = [
        {"chain": lig["chain"], "residue_name": lig["residue_name"],
         "residue_seqid": lig["residue_seqid"], "atom_count": lig["atom_count"]}
        for lig in ligands
    ]

    # 分辨率提取（PDB REMARK 2 / mmCIF refinement）
    resolution = _extract_resolution(st)
    if resolution:
        report["resolution"] = resolution
        if resolution > 4.0:
            report["issues"].append(f"低分辨率 ({resolution:.1f} Å)；对接结果可能不可靠")

    if atom_count == 0:
        report["issues"].append("结构不含任何原子")
    if not chain_info:
        report["issues"].append("结构不含任何链")
    has_amino_acid = False
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name.strip().upper() in _STANDARD_AMINO_ACIDS:
                    has_amino_acid = True
                    break
            if has_amino_acid:
                break
        if has_amino_acid:
            break
    if not has_amino_acid:
        report["issues"].append("未检测到氨基酸残基；可能不是蛋白结构")

    return report


# ── 配体检测 ─────────────────────────────────────────────────────────

_SOLVENT_AND_IONS = {
    "HOH", "DOD", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE",
    "SO4", "PO4", "GOL", "EDO", "ACT", "EPE", "PEG", "MPD", "BME", "DTT",
    "TRS", "HEPES", "MES", "PBS",
}

_STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "PYL", "SEC", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN", "ACE",
    "NME", "MSE", "MLY", "MLZ", "MLE",
}

_NUCLEIC_ACIDS = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU"}


def find_ligands(structure: gemmi.Structure, model_index: int = 0) -> list[dict[str, Any]]:
    """检测非蛋白小分子配体。

    Returns:
        配体列表，每个元素包含 chain, residue_name, residue_seqid, atom_count, center.
    """
    if model_index >= len(structure):
        return []
    model = structure[model_index]
    ligands: list[dict[str, Any]] = []
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            name = residue.name.strip().upper()
            if not name or name in _SOLVENT_AND_IONS or name in _STANDARD_AMINO_ACIDS:
                continue
            if name in _NUCLEIC_ACIDS:
                continue
            atoms = list(residue)
            if not atoms:
                continue
            center = [0.0, 0.0, 0.0]
            for a in atoms:
                center[0] += a.pos.x
                center[1] += a.pos.y
                center[2] += a.pos.z
            n = len(atoms)
            ligands.append({
                "chain": chain.name,
                "residue_name": residue.name,
                "residue_seqid": str(residue.seqid.num) if residue.seqid.num is not None else "",
                "atom_count": n,
                "center": [round(c / n, 4) for c in center],
            })
    return ligands


# ── 链抽取 ───────────────────────────────────────────────────────────

def extract_chain(
    structure: gemmi.Structure,
    chain_name: str,
    model_index: int = 0,
) -> gemmi.Structure | None:
    """从结构中提取指定链，返回仅包含该链的新 Structure。

    用于分离蛋白链和配体链，方便下游处理。
    """
    if model_index >= len(structure):
        return None
    src_model = structure[model_index]
    src_chain = src_model.find_chain(chain_name)
    if not src_chain:
        return None
    new_st = gemmi.Structure()
    new_model = gemmi.Model("1")
    new_chain = src_chain.clone()
    new_model.add_chain(new_chain)
    new_st.add_model(new_model)
    return new_st


# ── 口袋残基搜索 ─────────────────────────────────────────────────────

def find_residues_around(
    structure: gemmi.Structure,
    center: tuple[float, float, float],
    radius: float = 8.0,
    model_index: int = 0,
) -> list[dict[str, Any]]:
    """通过 Gemmi NeighborSearch 查找口袋周围的残基。

    Args:
        structure: Gemmi 结构对象
        center: 搜索中心 (x, y, z)
        radius: 搜索半径 (Å)
        model_index: 模型索引

    Returns:
        残基列表，每个含 chain, res_name, res_seqid, atom_count, min_distance
    """
    if model_index >= len(structure):
        return []
    model = structure[model_index]
    ns = gemmi.NeighborSearch(structure, max(radius, 6.0))
    ns.populate(include_h=False)

    center_pos = gemmi.Position(*center)
    marks = ns.find_atoms(center_pos, radius=radius)

    residues: dict[tuple[str, str, str], dict[str, Any]] = {}
    for mark in marks:
        cra = mark.to_cra(model)
        chain_name = cra.chain.name
        res_name = cra.residue.name
        res_seqid = str(cra.residue.seqid.num) if cra.residue.seqid.num is not None else ""
        key = (chain_name, res_name, res_seqid)
        dist = mark.pos.dist(center_pos)
        if key not in residues:
            residues[key] = {
                "chain": chain_name,
                "residue_name": res_name,
                "residue_seqid": res_seqid,
                "atom_count": 0,
                "min_distance": float(dist),
            }
            residues[key]["atom_count"] += 1
        else:
            residues[key]["atom_count"] += 1
            if dist < residues[key]["min_distance"]:
                residues[key]["min_distance"] = float(dist)
    return sorted(residues.values(), key=lambda r: r["min_distance"])


# ── 格式转换 ─────────────────────────────────────────────────────────

def convert_format(
    input_path: str | Path,
    output_path: str | Path,
    output_format: str = "pdb",
) -> Path:
    """将 PDB ↔ mmCIF 互转。

    Args:
        input_path: 输入文件
        output_path: 输出文件
        output_format: "pdb" 或 "cif"

    Returns:
        输出文件路径
    """
    st = read_structure(input_path)
    out = Path(output_path)
    if output_format.lower() in {"pdb", ".pdb"}:
        st.write_pdb(str(out))
    elif output_format.lower() in {"cif", "mmcif", ".cif", ".mmcif"}:
        st.make_mmcif_document().write_file(str(out))
    else:
        raise ValueError(f"不支持的输出格式: {output_format}（需要 pdb 或 cif）")
    return out


def structure_summary(structure: gemmi.Structure) -> str:
    """生成结构的人类可读摘要。"""
    lines = []
    lines.append(f"Models: {len(structure)}")
    for i, model in enumerate(structure):
        lines.append(f"  Model {i + 1}:")
        for chain in model:
            residues = list(chain)
            lines.append(f"    Chain {chain.name}: {len(residues)} residues")
        ligands = find_ligands(structure, model_index=i)
        if ligands:
            lig_names = sorted(set(lig["residue_name"] for lig in ligands))
            lines.append(f"    Ligands: {', '.join(lig_names)}")
    resolution = _extract_resolution(structure)
    if resolution:
        lines.append(f"Resolution: {resolution:.2f} Å")
    return "\n".join(lines)


# ── 内部辅助 ─────────────────────────────────────────────────────────

def _extract_resolution(structure: gemmi.Structure) -> float | None:
    """从 PDB REMARK 2 或 mmCIF 中提取分辨率。"""
    # 通过 raw_remarks 查找
    for remark in structure.raw_remarks:
        if "RESOLUTION" in remark.upper():
            import re
            m = re.search(r"RESOLUTION[.\s]*(\d+\.?\d*)", remark, re.I)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    # mmCIF 方式
    try:
        for model in structure:
            if hasattr(model, "resolution") and model.resolution:
                return float(model.resolution)
    except Exception:
        pass
    return None


def _find_atom_in_structure(structure: gemmi.Structure, chain_name: str,
                            res_name: str, res_seqid: str, atom_name: str) -> gemmi.Atom | None:
    """在结构中查找单个原子。"""
    for model in structure:
        chain = model.find_chain(chain_name)
        if not chain:
            continue
        for residue in chain:
            if residue.name.strip().upper() != res_name.strip().upper():
                continue
            if str(residue.seqid.num) != str(res_seqid):
                continue
            for atom in residue:
                if atom.name.strip().upper() == atom_name.strip().upper():
                    return atom
    return None
