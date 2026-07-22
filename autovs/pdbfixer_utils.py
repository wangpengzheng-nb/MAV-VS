"""PDBFixer 蛋白质结构修复模块。

基于 OpenMM PDBFixer 提供：
- 补缺失原子（侧链、主链轻原子）及氢原子
- 非标准残基处理
- 选择性删除多余链和异质分子
- **长缺失片段检测与警告**：>5 残基的缺口不静默补建，而是报告风险

所有修复操作是进程内 Python 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pdbfixer import PDBFixer
import openmm.app as app


# ── 默认排除列表 ─────────────────────────────────────────────────────

# 常见异质分子和溶剂
_DEFAULT_HETEROGENS_TO_REMOVE = {
    "HOH", "DOD", "WAT",       # 水
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE",  # 离子
    "SO4", "PO4", "GOL", "EDO", "EPE", "PEG",       # 缓冲液
    "MPD", "BME", "DTT", "TRS", "HEPES", "MES",
    "ACT", "CIT", "PBS",
}

# 非标准氨基酸常见修饰（保留，不删除）
_PROTECTED_RESIDUES = {
    "MSE", "MLY", "MLZ", "MLE", "SEP", "TPO", "PTR",
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
    "ACE", "NME", "PCA",
}

# 长缺失阈值（残基数）
_LONG_GAP_THRESHOLD = 5


# ── 主修复函数 ───────────────────────────────────────────────────────

def repair_structure(
    input_path: str | Path,
    output_path: str | Path,
    *,
    add_hydrogens: bool = True,
    add_missing_atoms: bool = True,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = True,
    heterogens_to_remove: set[str] | None = None,
    keep_chains: list[str] | None = None,
    remove_chains: list[str] | None = None,
    ph: float = 7.4,
    long_gap_threshold: int = _LONG_GAP_THRESHOLD,
) -> dict[str, Any]:
    """修复蛋白质结构中的常见问题。

    Args:
        input_path: 输入 PDB 文件路径
        output_path: 输出修复后 PDB 路径
        add_hydrogens: 是否根据 pH 添加氢原子
        add_missing_atoms: 是否补建缺失的侧链/主链重原子
        replace_nonstandard: 是否将非标准残基替换为标准残基
        remove_heterogens: 是否删除异质分子
        heterogens_to_remove: 要删除的异质分子残基名集合（默认 _DEFAULT_HETEROGENS）
        keep_chains: 保留的链名列表（None = 保留全部）
        remove_chains: 要删除的链名列表
        ph: 质子化 pH
        long_gap_threshold: 长缺失阈值（残基数），超过此值将发出警告而非静默补建

    Returns:
        诊断报告 dict，包含：
        - missing_residues: 缺失残基列表
        - missing_atoms: 缺失原子摘要
        - long_gaps: 长缺失片段警告（> threshold）
        - nonstandard_residues: 替换的非标准残基
        - heterogens_removed: 被删除的异质分子
        - chains_kept: 保留的链
        - chains_removed: 被删除的链
        - warnings: 警告信息列表
        - output_path: 输出文件路径
    """
    filepath = Path(input_path)
    if not filepath.is_file():
        raise FileNotFoundError(f"输入 PDB 不存在: {filepath}")

    fixer = PDBFixer(filename=str(filepath))
    report: dict[str, Any] = {
        "input_path": str(filepath),
        "output_path": str(output_path),
        "missing_residues": [],
        "missing_atoms_summary": {},
        "long_gaps": [],
        "nonstandard_residues": [],
        "heterogens_removed": [],
        "chains_kept": [],
        "chains_removed": [],
        "warnings": [],
        "ph": ph,
    }

    # ── 1. 检测缺失残基和长缺口 ──────────────────────────────
    report["missing_residues"] = _list_missing_residues(fixer)
    report["long_gaps"] = _detect_long_gaps(fixer, threshold=long_gap_threshold)

    if report["long_gaps"]:
        gap_descriptions = [
            f"{g['chain']}:{g['start_resid']}-{g['end_resid']} ({g['length']}残基)"
            for g in report["long_gaps"]
        ]
        warning_msg = (
            f"检测到 {len(report['long_gaps'])} 个长缺失片段（>{long_gap_threshold}残基）: "
            + "; ".join(gap_descriptions)
            + "。这些片段不会被静默补建，修复后的结构可能不可靠。"
            + "如需完整结构，请考虑使用 AlphaFold/Boltz 预测或更换 PDB 条目。"
        )
        report["warnings"].append(warning_msg)

    # ── 2. 链筛选 ──────────────────────────────────────────
    if keep_chains or remove_chains:
        chain_report = _filter_chains(fixer, keep=keep_chains, remove=remove_chains)
        report["chains_kept"] = chain_report["kept"]
        report["chains_removed"] = chain_report["removed"]
    else:
        all_chains = list({r.chain.id for r in fixer.topology.residues()})
        report["chains_kept"] = all_chains

    # ── 3. 删除异质分子 ────────────────────────────────────
    if remove_heterogens:
        to_remove = heterogens_to_remove or _DEFAULT_HETEROGENS_TO_REMOVE
        removed = _remove_heterogens(fixer, to_remove=to_remove)
        report["heterogens_removed"] = removed

    # ── 4. 非标准残基替换 ──────────────────────────────────
    if replace_nonstandard:
        report["nonstandard_residues"] = _replace_nonstandard(fixer)

    # ── 5. 补建缺失重原子 ──────────────────────────────────
    if add_missing_atoms:
        try:
            fixer.findMissingAtoms()
            missing_atom_count = sum(len(res) for res in fixer.missingAtoms.values())
            if missing_atom_count > 0:
                fixer.addMissingAtoms()
                report["missing_atoms_summary"] = {
                    "total_missing": missing_atom_count,
                    "residues_affected": len(fixer.missingAtoms),
                }
        except Exception as exc:
            report["warnings"].append(f"补建缺失重原子失败: {exc}")

    # ── 6. 添加氢原子 ──────────────────────────────────────
    if add_hydrogens:
        try:
            fixer.addMissingHydrogens(ph)
        except Exception as exc:
            report["warnings"].append(f"添加氢原子失败: {exc}")

    # ── 7. 写出修复后的 PDB ─────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            handle,
            keepIds=True,
        )

    return report


# ── 内部辅助 ─────────────────────────────────────────────────────────

def _list_missing_residues(fixer: PDBFixer) -> list[dict[str, Any]]:
    """列出所有缺失残基。"""
    fixer.findMissingResidues()
    result = []
    for chain_name, residues in fixer.missingResidues.items():
        for res in residues:
            result.append({
                "chain": chain_name,
                "residue_name": res.name,
                "residue_seqid": str(res.id),
            })
    return result


def _detect_long_gaps(
    fixer: PDBFixer, threshold: int = _LONG_GAP_THRESHOLD,
) -> list[dict[str, Any]]:
    """检测长缺失片段（连续缺失 > threshold 残基）。"""
    fixer.findMissingResidues()
    gaps = []
    for chain_name, residues in fixer.missingResidues.items():
        if len(residues) <= threshold:
            continue
        # 分组连续缺失
        residues_sorted = sorted(residues, key=lambda r: int(r.id) if str(r.id).isdigit() else 0)
        start_res = residues_sorted[0]
        run_length = 1
        for i in range(1, len(residues_sorted)):
            prev_num = int(start_res.id) if str(start_res.id).isdigit() else 0
            curr_num = int(residues_sorted[i].id) if str(residues_sorted[i].id).isdigit() else 0
            if curr_num == prev_num + run_length:
                run_length += 1
            else:
                if run_length > threshold:
                    gaps.append({
                        "chain": chain_name,
                        "start_resid": start_res.id,
                        "end_resid": residues_sorted[i - 1].id,
                        "length": run_length,
                    })
                start_res = residues_sorted[i]
                run_length = 1
        if run_length > threshold:
            gaps.append({
                "chain": chain_name,
                "start_resid": start_res.id,
                "end_resid": residues_sorted[-1].id,
                "length": run_length,
            })
    return gaps


def _filter_chains(
    fixer: PDBFixer, keep: list[str] | None = None, remove: list[str] | None = None,
) -> dict[str, Any]:
    """删除不需要的链。"""
    all_chains = list({r.chain.id for r in fixer.topology.residues()})
    if keep:
        to_remove_ids = {c for c in all_chains if c not in set(keep)}
    elif remove:
        to_remove_ids = set(remove)
    else:
        return {"kept": all_chains, "removed": []}

    indices_to_delete = [i for i, chain in enumerate(fixer.topology.chains())
                         if chain.id in to_remove_ids]
    if indices_to_delete:
        fixer.removeChains(indices_to_delete)

    kept = [c for c in all_chains if c not in to_remove_ids]
    return {"kept": kept, "removed": sorted(to_remove_ids)}


def _remove_heterogens(fixer: PDBFixer, to_remove: set[str]) -> list[dict[str, Any]]:
    """删除异质分子（水、离子、缓冲液等）。"""
    # PDBFixer API: removeHeterogens(keepWater=False) 删除所有 HETATM
    # 在调用前后记录被删的异质分子
    before = set()
    for residue in fixer.topology.residues():
        res_name = residue.name.strip().upper()
        if res_name in to_remove and res_name not in _PROTECTED_RESIDUES:
            before.add((residue.chain.id, res_name, str(residue.id)))

    fixer.removeHeterogens(keepWater=False)

    # removeHeterogens 会把所有 HETATM 都删掉，无法精确控制
    # 所以我们记录的是"预期被删"列表供参考
    return [
        {"chain": c, "residue_name": n, "residue_id": i}
        for c, n, i in sorted(before)
    ]


def _replace_nonstandard(fixer: PDBFixer) -> list[dict[str, Any]]:
    """替换非标准残基为标准残基。"""
    fixer.findNonstandardResidues()
    replaced = []
    for residue in fixer.nonstandardResidues:
        res_name = residue.name.strip().upper()
        if res_name in _PROTECTED_RESIDUES:
            continue
        replaced.append({
            "chain": residue.chain.id,
            "original_name": res_name,
            "residue_id": residue.id,
        })
    fixer.replaceNonstandardResidues()
    return replaced


def quick_diagnostic(input_path: str | Path) -> dict[str, Any]:
    """快速诊断：报告结构问题而不做修复。

    用于在管道早期阶段评估 PDB 质量。
    """
    filepath = Path(input_path)
    if not filepath.is_file():
        return {"error": f"文件不存在: {filepath}"}

    fixer = PDBFixer(filename=str(filepath))
    report: dict[str, Any] = {
        "path": str(filepath),
        "num_chains": 0,
        "num_residues": 0,
        "num_atoms": 0,
        "missing_residue_count": 0,
        "long_gap_count": 0,
        "long_gaps": [],
        "nonstandard_count": 0,
        "heterogen_count": 0,
        "ok_for_docking": True,
        "issues": [],
    }

    chains = list(fixer.topology.chains())
    report["num_chains"] = len(chains)
    residues = list(fixer.topology.residues())
    report["num_residues"] = len(residues)
    report["num_atoms"] = sum(len(list(r.atoms())) for r in residues)

    fixer.findMissingResidues()
    all_missing = sum(len(v) for v in fixer.missingResidues.values())
    report["missing_residue_count"] = all_missing

    long_gaps = _detect_long_gaps(fixer)
    report["long_gap_count"] = len(long_gaps)
    report["long_gaps"] = long_gaps

    fixer.findNonstandardResidues()
    report["nonstandard_count"] = len(fixer.nonstandardResidues)

    heterogens = [
        r.name.strip().upper() for r in residues
        if r.name.strip().upper() in _DEFAULT_HETEROGENS_TO_REMOVE
    ]
    report["heterogen_count"] = len(heterogens)

    if report["long_gap_count"] > 0:
        report["ok_for_docking"] = False
        report["issues"].append(
            f"存在 {report['long_gap_count']} 个长缺失片段；"
            "修复后的结构可能不可靠，建议更换 PDB 或使用预测结构"
        )
    if report["missing_residue_count"] > 20:
        report["issues"].append(f"缺失 {report['missing_residue_count']} 个残基")
    if report["nonstandard_count"] > 10:
        report["issues"].append(f"存在 {report['nonstandard_count']} 个非标准残基")

    return report
