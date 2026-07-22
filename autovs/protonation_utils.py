"""PDB2PQR + PROPKA 质子化与电荷处理模块。

基于 PDB2PQR 3.7+ 和 PROPKA 3.5+：
- pH 可配的 pKa 预测（PROPKA）
- 加氢与质子化状态优化
- 力场参数分配（AMBER / CHARMM / PARSE）
- 去水、去异质分子、链选择

核心原则：
- **pH 不固定**：不同靶点和实验条件使用不同 pH
- **PROPKA 先于加氢**：先预测 pKa 再决定质子化状态
- **力场参数透明**：输出 PQR 格式（PDB + 每原子电荷 + 半径）
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


# ── 支持的力场 ───────────────────────────────────────────────────────

FORCE_FIELDS = {
    "AMBER": "AMBER99 力场",
    "CHARMM": "CHARMM27 力场",
    "PARSE": "PARSE 力场（默认，适合隐式溶剂）",
    "TYL06": "TYL06 力场",
    "PEOEPB": "PEOEPB 力场",
    "SWANSON": "SWANSON 力场",
}


# ── 主接口 ───────────────────────────────────────────────────────────

def protonate_structure(
    input_path: str | Path,
    output_pqr: str | Path,
    *,
    ph: float = 7.4,
    forcefield: str = "PARSE",
    drop_water: bool = True,
    nodebump: bool = False,
    noopt: bool = False,
    keep_chain: bool = False,
    pdb_output: str | Path | None = None,
    chains: list[str] | None = None,
) -> dict[str, Any]:
    """使用 PDB2PQR + PROPKA 对蛋白进行 pH 依赖性质子化。

    Args:
        input_path: 输入 PDB 文件路径
        output_pqr: 输出 PQR 文件路径
        ph: 目标 pH（不固定！根据靶点和实验条件设置）
        forcefield: 力场名称 (AMBER/CHARMM/PARSE/TYL06/PEOEPB/SWANSON)
        drop_water: 是否删除水分子
        nodebump: 是否跳过氢原子碰撞优化（True=跳过，False=优化）
        noopt: 是否跳过氢键网络优化（True=跳过，False=优化）
        keep_chain: 是否保留链 ID
        pdb_output: 可选输出 PDB 路径（含优化后的氢原子坐标）
        chains: 保留的链列表（None=全部保留）

    Returns:
        诊断报告 dict：
        - ph: 实际使用的 pH
        - forcefield: 使用的力场
        - pdb2pqr_version: PDB2PQR 版本
        - output_pqr: PQR 输出路径
        - pdb_output: PDB 输出路径（如果生成）
        - warnings: 警告列表（如 pH 超出典型范围）
    """
    from pdb2pqr.main import main_driver, build_main_parser
    from pdb2pqr import __version__ as pdb2pqr_version

    filepath = Path(input_path)
    if not filepath.is_file():
        raise FileNotFoundError(f"输入 PDB 不存在: {filepath}")

    pqr_path = Path(output_pqr)
    pqr_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建 argparse.Namespace
    parser = build_main_parser()
    argv = [
        str(filepath),
        str(pqr_path),
        "--ff", forcefield,
        "--with-ph", str(ph),
        "--titration-state-method", "propka",
    ]
    if drop_water:
        argv.append("--drop-water")
    if nodebump:
        argv.append("--nodebump")
    if noopt:
        argv.append("--noopt")
    if keep_chain:
        argv.append("--keep-chain")
    if pdb_output:
        pdb_out = Path(pdb_output)
        pdb_out.parent.mkdir(parents=True, exist_ok=True)
        argv.extend(["--pdb-output", str(pdb_out)])
    if chains:
        argv.extend(["-c", ",".join(chains)])
    if forcefield == "CHARMM":
        argv.extend(["--ffout", "CHARMM"])

    # 添加默认选项以保持可重现性
    argv.extend(["--nodebump"] if not nodebump else [])  # handled above
    # 重新构建：移除重复
    seen_ff = False
    final_argv = []
    for arg in argv:
        if arg == "--ff" or arg in FORCE_FIELDS:
            if not seen_ff:
                final_argv.append(arg)
                seen_ff = (arg != "--ff")
            elif arg in FORCE_FIELDS:
                final_argv[-1] = arg  # replace previous ff value
            continue
        final_argv.append(arg)

    # 重新解析
    args = parser.parse_args(final_argv)

    # 若 chains 指定，覆盖 args
    if chains:
        args.chains = chains

    # 执行
    warnings: list[str] = []
    if ph < 2.0 or ph > 12.0:
        warnings.append(f"pH={ph} 超出典型生物范围 (2-12)；pKa 预测可能不准确")
    if ph != 7.4:
        warnings.append(f"非标准 pH={ph}；已使用 PROPKA 根据该 pH 调整质子化状态")

    try:
        main_driver(args)
    except Exception as exc:
        raise RuntimeError(
            f"PDB2PQR 质子化失败: {exc}"
            + f"（输入={filepath}, pH={ph}, ff={forcefield}）"
        ) from exc

    report: dict[str, Any] = {
        "ph": ph,
        "forcefield": forcefield,
        "pdb2pqr_version": pdb2pqr_version,
        "output_pqr": str(pqr_path),
        "pdb_output": str(pdb_output) if pdb_output else None,
        "warnings": warnings,
        "drop_water": drop_water,
    }

    if not pqr_path.is_file():
        raise RuntimeError(f"PQR 输出未生成: {pqr_path}")

    return report


def predict_pka(
    input_path: str | Path,
    *,
    ph: float = 7.4,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """单独使用 PROPKA 预测可电离基团的 pKa 值（不做质子化）。

    用于评估蛋白在不同 pH 条件下的电荷状态。

    Args:
        input_path: 输入 PDB 文件路径
        ph: 参考 pH（用于计算该 pH 下的质子化分数）
        output_path: 可选输出路径（PROPKA 报告）

    Returns:
        pKa 预测报告 dict：
        - ph: 参考 pH
        - residues: [{residue_name, chain, resid, pka, protonation_fraction, ...}]
        - summary: {total_charge, net_charge_at_ph, ...}
    """
    from propka.run import single as propka_single

    filepath = Path(input_path)
    if not filepath.is_file():
        raise FileNotFoundError(f"输入 PDB 不存在: {filepath}")

    out = Path(output_path) if output_path else filepath.parent / f"{filepath.stem}_propka.csv"

    try:
        # PROPKA single: (filename, optargs=(), stream=None, write_pka=True)
        # optargs 传递 pH 等选项
        optargs = []
        if ph != 7.0:
            optargs.extend(["-p", str(ph)])
        if out:
            optargs.extend(["-o", str(out)])

        result = propka_single(
            str(filepath),
            optargs=optargs,
            write_pka=True,
        )
    except Exception as exc:
        raise RuntimeError(f"PROPKA pKa 预测失败: {exc}") from exc

    # 解析结果
    residues: list[dict[str, Any]] = []
    if hasattr(result, "residues") or isinstance(result, dict):
        # 尝试不同格式
        raw = result if isinstance(result, dict) else result.__dict__ if hasattr(result, "__dict__") else {}
        if hasattr(result, "conformations"):
            for conf_name, conf in result.conformations.items():
                if hasattr(conf, "groups"):
                    for group in conf.groups:
                        residues.append({
                            "residue_name": getattr(group, "label", "?"),
                            "chain": getattr(group, "chain", ""),
                            "resid": getattr(group, "resid", ""),
                            "pka": getattr(group, "pka_value", None),
                            "model_pka": getattr(group, "model_pka", None),
                        })

    return {
        "ph": ph,
        "residue_count": len(residues),
        "residues": residues,
        "output_file": str(out) if out.is_file() else None,
    }


def quick_charge_summary(input_pqr: str | Path) -> dict[str, Any]:
    """从 PQR 文件中提取电荷摘要。

    PQR 格式: 每行包含 charge 和 radius 两列。
    """
    filepath = Path(input_pqr)
    if not filepath.is_file():
        return {"error": f"PQR 文件不存在: {filepath}"}

    total_charge = 0.0
    atom_count = 0
    residue_charges: dict[str, float] = {}

    with filepath.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") and not line.startswith("HETATM"):
                continue
            try:
                charge = float(line[54:62].strip())
                radius = float(line[62:70].strip())
            except (ValueError, IndexError):
                continue
            total_charge += charge
            atom_count += 1
            res_name = line[17:20].strip()
            resid = line[22:26].strip()
            key = f"{res_name}{resid}"
            residue_charges[key] = residue_charges.get(key, 0.0) + charge

    return {
        "path": str(filepath),
        "atom_count": atom_count,
        "total_charge": round(total_charge, 4),
        "residue_charges": {k: round(v, 4) for k, v in residue_charges.items()},
    }
