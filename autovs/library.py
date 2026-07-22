from __future__ import annotations

import hashlib
import json
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rdkit import Chem, RDLogger

from autovs.security import sha256_file


RDLogger.DisableLog("rdApp.*")
MOLECULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
HEADER_IDS = {"id", "molecule_id", "mol_id", "source_id", "name"}
HEADER_SMILES = {"smiles", "smi", "canonical_smiles"}
STRICT_SMI_FORMAT = "molecule_id<TAB>SMILES (UTF-8, no header)"


@dataclass(frozen=True)
class SmiRecord:
    line_number: int
    molecule_id: str
    smiles: str


class SmiFormatError(ValueError):
    def __init__(self, line_number: int, error_type: str, detail: str, example: str = ""):
        self.line_number = line_number
        self.error_type = error_type
        self.detail = detail
        self.example = example[:240]
        suffix = f"; content={self.example!r}" if self.example else ""
        super().__init__(f"SMI format error at line {line_number} [{error_type}]: {detail}{suffix}")

    def as_dict(self) -> dict[str, object]:
        return {
            "line_number": self.line_number,
            "error_type": self.error_type,
            "detail": self.detail,
            "example": self.example,
            "expected_format": STRICT_SMI_FORMAT,
        }


def iter_strict_smi(path: Path, *, max_molecules: int | None = None) -> Iterator[SmiRecord]:
    if path.suffix.lower() not in {".smi", ".smiles"}:
        raise SmiFormatError(0, "unsupported_extension", "only .smi and .smiles files are accepted", path.name)
    line_number = 0
    try:
        with path.open(encoding="utf-8", errors="strict", newline=None) as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    raise SmiFormatError(line_number, "blank_line", "blank or whitespace-only records are forbidden")
                if line.lstrip().startswith("#"):
                    raise SmiFormatError(line_number, "comment_line", "comment records are forbidden", line)
                if line.count("\t") != 1:
                    raise SmiFormatError(line_number, "column_count", "each record must contain exactly two TAB-separated columns", line)
                molecule_id, smiles = line.split("\t")
                if not molecule_id or not smiles:
                    raise SmiFormatError(line_number, "empty_field", "molecule ID and SMILES must both be non-empty", line)
                if molecule_id != molecule_id.strip() or smiles != smiles.strip():
                    raise SmiFormatError(line_number, "surrounding_whitespace", "fields must not contain surrounding whitespace", line)
                if molecule_id.lower() in HEADER_IDS and smiles.lower() in HEADER_SMILES:
                    raise SmiFormatError(line_number, "header_forbidden", "SMI files must not contain a header", line)
                if not MOLECULE_ID_RE.fullmatch(molecule_id):
                    raise SmiFormatError(line_number, "invalid_molecule_id", "molecule ID must match ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$", line)
                if max_molecules is not None and line_number > max_molecules:
                    raise SmiFormatError(line_number, "library_limit", f"library exceeds configured limit of {max_molecules:,} molecules", line)
                yield SmiRecord(line_number, molecule_id, smiles)
    except UnicodeDecodeError as exc:
        raise SmiFormatError(max(1, line_number), "invalid_utf8", "file must be valid UTF-8") from exc
    if line_number == 0:
        raise SmiFormatError(1, "empty_file", "the molecular library is empty")


def validate_smi_structure(path: Path, *, max_molecules: int | None = None) -> int:
    return sum(1 for _ in iter_strict_smi(path, max_molecules=max_molecules))


def structure_id(canonical_smiles: str) -> str:
    return "mol_" + hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:16]


def normalize_smi_library(input_path: Path, output_dir: Path, *, max_molecules: int = 1_000_000,
                          source: str = "user", version: str | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "normalized_library.smi"
    rejected_path = output_dir / "library_rejected.tsv"
    validation_path = output_dir / "library_validation.json"
    rejected: list[dict[str, object]] = []
    accepted_ids: set[str] = set()
    canonical_owner: dict[str, str] = {}
    total_records = validate_smi_structure(input_path, max_molecules=max_molecules)
    accepted_count = 0
    with normalized_path.open("w", encoding="utf-8") as normalized_handle:
        for record in iter_strict_smi(input_path, max_molecules=max_molecules):
            mol = Chem.MolFromSmiles(record.smiles)
            was_repaired = False
            output_smiles = record.smiles

            if mol is None:
                # ── RDKit 自动修复 ──
                repaired = repair_smiles(record.smiles)
                if repaired is not None:
                    mol = Chem.MolFromSmiles(repaired)
                    if mol is None:
                        # repair_smiles 可能返回仅 sanitize=False 可读的原始 SMILES
                        mol = Chem.MolFromSmiles(repaired, sanitize=False)
                        if mol is not None:
                            try:
                                mol.UpdatePropertyCache(strict=False)
                                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                                                 catchErrors=True)
                            except Exception:
                                pass
                    if mol is not None:
                        was_repaired = True
                        output_smiles = repaired
                    else:
                        rejected.append({"line_number": record.line_number, "molecule_id": record.molecule_id,
                                         "smiles": record.smiles, "reason": "unrepairable_smiles"})
                        continue
                else:
                    rejected.append({"line_number": record.line_number, "molecule_id": record.molecule_id,
                                     "smiles": record.smiles, "reason": "invalid_smiles"})
                    continue

            try:
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            except Exception:
                canonical = output_smiles
            if record.molecule_id in accepted_ids:
                rejected.append({"line_number": record.line_number, "molecule_id": record.molecule_id,
                                 "smiles": record.smiles, "reason": "duplicate_molecule_id"})
                continue
            if canonical in canonical_owner:
                rejected.append({"line_number": record.line_number, "molecule_id": record.molecule_id,
                                 "smiles": record.smiles, "reason": f"duplicate_structure_of:{canonical_owner[canonical]}"})
                continue
            accepted_ids.add(record.molecule_id)
            canonical_owner[canonical] = record.molecule_id
            normalized_handle.write(f"{record.molecule_id}\t{canonical}\n")
            accepted_count += 1
    if not accepted_count:
        raise ValueError("no valid molecules remain after strict SMI validation")
    rejected_path.write_text(
        "line_number\tmolecule_id\tsmiles\treason\n" +
        "".join(f"{row['line_number']}\t{row['molecule_id']}\t{row['smiles']}\t{row['reason']}\n" for row in rejected),
        encoding="utf-8",
    )
    result = {
        "format": "strict_smi_v1", "source": source, "version": version,
        "input_path": str(input_path), "input_sha256": sha256_file(input_path),
        "normalized_path": str(normalized_path), "normalized_sha256": sha256_file(normalized_path),
        "total_records": total_records, "accepted_records": accepted_count,
        "quarantined_records": len(rejected), "rejection_reasons": _reason_counts(rejected),
    }
    validation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "normalized_library": normalized_path, "rejected": rejected_path,
            "validation": validation_path}


def verify_default_library(path: Path, expected_sha256: str, expected_count: int) -> dict[str, object]:
    if not path.is_file():
        return {"status": "unavailable", "reason": f"default library missing: {path}"}
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        return {"status": "unavailable", "reason": "default library checksum mismatch",
                "expected_sha256": expected_sha256, "actual_sha256": actual_sha}
    try:
        count = validate_smi_structure(path, max_molecules=expected_count)
    except (SmiFormatError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    if count != expected_count:
        return {"status": "unavailable", "reason": f"default library count mismatch: {count} != {expected_count}"}
    return {"status": "available", "path": str(path), "sha256": actual_sha, "molecule_count": count}


def migrate_legacy_library(input_path: Path, output_path: Path) -> Path:
    """One-time compatibility converter for already-persisted pre-v1 tasks only."""
    try:
        validate_smi_structure(input_path)
        return input_path
    except (SmiFormatError, UnicodeDecodeError):
        pass
    rows: list[tuple[str, str]] = []
    suffix = input_path.suffix.lower()
    if suffix in {".smi", ".smiles", ".txt"}:
        for index, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            fields = line.split()
            if fields:
                rows.append((fields[1] if len(fields) > 1 else f"row_{index:09d}", fields[0]))
    elif suffix == ".csv":
        with input_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            smiles_col = next((name for name in names if name.lower() in {"smiles", "canonical_smiles"}), None)
            id_col = next((name for name in names if name.lower() in {"source_id", "mol_id", "id", "name"}), None)
            if not smiles_col:
                raise ValueError("legacy CSV has no SMILES column")
            for index, row in enumerate(reader, 1):
                rows.append((row.get(id_col, "") if id_col else f"row_{index:09d}", row.get(smiles_col, "")))
    elif suffix == ".sdf":
        for index, mol in enumerate(Chem.SDMolSupplier(str(input_path), removeHs=False), 1):
            if mol is None:
                continue
            molecule_id = (mol.GetProp("source_id") if mol.HasProp("source_id") else
                           mol.GetProp("_Name") if mol.HasProp("_Name") else f"row_{index:09d}")
            rows.append((molecule_id, Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)))
    else:
        raise ValueError(f"legacy task library format cannot be migrated: {suffix}")
    if not rows:
        raise ValueError("legacy task library contains no records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (raw_id, smiles) in enumerate(rows, 1):
            cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(raw_id)).strip("_.:-")[:128]
            molecule_id = cleaned or f"row_{index:09d}"
            handle.write(f"{molecule_id}\t{smiles}\n")
    validate_smi_structure(output_path)
    return output_path


def _reason_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"]).split(":", 1)[0]
        counts[reason] = counts.get(reason, 0) + 1
    return counts


# ── SMILES 修复策略 ──────────────────────────────────────────────

def repair_smiles(raw_smiles: str) -> str | None:
    """尝试用多种 RDKit 策略修复格式不正确的 SMILES。

    返回修复后可用的 SMILES（优先 canonical），无法修复则返回 None。
    对于部分可读但无法 canonicalize 的分子，返回原始 SMILES。
    """
    # 策略1: 直接解析（标准路径）→ canonicalize
    try:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            verify = Chem.MolFromSmiles(canonical)
            if verify is not None:
                return canonical
    except Exception:
        pass

    # 策略2: sanitize=False 读取后逐步手动 sanitize
    mol_unsanitized = _try_unsanitized(raw_smiles)
    if mol_unsanitized is not None:
        try:
            canonical = Chem.MolToSmiles(mol_unsanitized, canonical=True, isomericSmiles=True)
            verify = Chem.MolFromSmiles(canonical)
            if verify is not None:
                return canonical
        except Exception:
            pass
        # 无法 canonicalize 但分子对象有效 → 返回原始 SMILES
        # （下游 prepare_library 会用 sanitize=False 处理）
        return raw_smiles

    # 策略3: sanitize=False 读取后直接 aromatize 往返
    fixed = _try_roundtrip(raw_smiles)
    if fixed is not None:
        return fixed

    # 策略4: 拆分断开的结构（如溶剂、盐），取最大有机片段
    fragment = _try_largest_fragment(raw_smiles)
    if fragment is not None:
        try:
            return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        except Exception:
            return raw_smiles

    # 策略5: 去掉所有显式氢标记后解析
    if "[H]" in raw_smiles:
        try:
            mol = Chem.MolFromSmiles(raw_smiles.replace("[H]", ""))
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            pass

    # 策略6: 移除手性标记后重试
    if "@" in raw_smiles:
        try:
            mol = Chem.MolFromSmiles(raw_smiles.replace("@@", "@").replace("@", ""))
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            pass

    return None


def _try_roundtrip(smiles: str):
    """用 sanitize=False 读取后直接输出 canonical SMILES，通过往返修复 kekulization。"""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        # 尝试用 aromatic SMILES 往返
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                         catchErrors=True)
        aromatic_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return Chem.MolFromSmiles(aromatic_smi)
    except Exception:
        return None


def _try_unsanitized(smiles: str):
    """用 sanitize=False 读取，然后逐步手动 sanitize。"""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        # 跳过 SANITIZE_KEKULIZE，先做其他 sanitize
        Chem.SanitizeMol(mol,
                         Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        # 最后单独尝试 kekulize
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass  # kekulize 失败也可以接受
        return mol
    except Exception:
        try:
            # 更保守的 sanitize
            mol2 = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol2 is None:
                return None
            mol2.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol2,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                             Chem.SanitizeFlags.SANITIZE_CLEANUP |
                             Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY |
                             Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                             catchErrors=True)
            return mol2
        except Exception:
            return None


def _try_largest_fragment(smiles: str):
    """分离混合结构（盐/溶剂），取最大有机片段。"""
    parts = smiles.split(".")
    if len(parts) <= 1:
        return None
    best_mol = None
    best_weight = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = Chem.MolFromSmiles(part)
        if m is None:
            continue
        from rdkit.Chem import Descriptors
        mw = Descriptors.MolWt(m)
        if mw > best_weight:
            best_mol = m
            best_weight = mw
    return best_mol


def repair_smi_library(input_path: Path, output_dir: Path, *,
                       max_molecules: int = 1_000_000,
                       source: str = "user") -> dict[str, object]:
    """用 RDKit 修复策略处理用户上传的分子库。

    对每个分子尝试 repair_smiles()，修复成功则输出 canonical SMILES，
    无法修复则丢弃（写入 rejected 列表）。

    Returns:
        dict with keys: repaired_library, repaired_path, repaired_sha256,
                        total_records, repaired_records, rejected_records, rejected
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = output_dir / "repaired_library.smi"
    rejected_path = output_dir / "repair_rejected.tsv"
    summary_path = output_dir / "repair_summary.json"

    rejected: list[dict[str, object]] = []
    accepted_ids: set[str] = set()
    canonical_owner: dict[str, str] = {}
    repaired_count = 0
    fixed_count = 0
    total_records = 0

    with repaired_path.open("w", encoding="utf-8") as out:
        for record in iter_strict_smi(input_path, max_molecules=max_molecules):
            total_records += 1
            original_smiles = record.smiles

            # 先尝试直接解析
            mol = Chem.MolFromSmiles(original_smiles)
            was_fixed = False
            output_smiles = original_smiles

            if mol is None:
                # 尝试修复
                repaired = repair_smiles(original_smiles)
                if repaired is None:
                    rejected.append({
                        "line_number": record.line_number,
                        "molecule_id": record.molecule_id,
                        "smiles": original_smiles,
                        "reason": "unrepairable_smiles",
                    })
                    continue

                was_fixed = True
                output_smiles = repaired
                mol = Chem.MolFromSmiles(repaired)

                if mol is None:
                    # repair_smiles 可能返回原始 SMILES（仅 sanitize=False 可读）
                    # 尝试用 sanitize=False 获取 mol 对象以进行 canonicalize
                    mol = Chem.MolFromSmiles(repaired, sanitize=False)
                    if mol is None:
                        rejected.append({
                            "line_number": record.line_number,
                            "molecule_id": record.molecule_id,
                            "smiles": original_smiles,
                            "reason": "repair_produced_invalid",
                        })
                        continue
                    # sanitize=False 读取成功，用 aromatic SMILES 作为 canonical
                    try:
                        mol.UpdatePropertyCache(strict=False)
                        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                                         catchErrors=True)
                        output_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    except Exception:
                        output_smiles = repaired  # 保持修复后的 SMILES

            try:
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            except Exception:
                canonical = output_smiles

            # 去重检查
            if record.molecule_id in accepted_ids:
                rejected.append({
                    "line_number": record.line_number,
                    "molecule_id": record.molecule_id,
                    "smiles": original_smiles,
                    "reason": "duplicate_molecule_id",
                })
                continue
            if canonical in canonical_owner:
                rejected.append({
                    "line_number": record.line_number,
                    "molecule_id": record.molecule_id,
                    "smiles": original_smiles,
                    "reason": f"duplicate_structure_of:{canonical_owner[canonical]}",
                })
                continue

            accepted_ids.add(record.molecule_id)
            canonical_owner[canonical] = record.molecule_id
            out.write(f"{record.molecule_id}\t{canonical}\n")
            repaired_count += 1
            if was_fixed:
                fixed_count += 1

    if not repaired_count:
        raise ValueError("分子库修复后无有效分子：所有 SMILES 均无法被 RDKit 解析或修复")

    # 写入 rejected 列表
    rejected_path.write_text(
        "line_number\tmolecule_id\tsmiles\treason\n" +
        "".join(f"{r['line_number']}\t{r['molecule_id']}\t{r['smiles']}\t{r['reason']}\n"
                for r in rejected),
        encoding="utf-8",
    )

    result = {
        "format": "strict_smi_v1",
        "source": source,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "repaired_path": str(repaired_path),
        "repaired_sha256": sha256_file(repaired_path),
        "total_records": total_records,
        "repaired_records": repaired_count,
        "fixed_records": fixed_count,
        "rejected_records": len(rejected),
        "rejected": str(rejected_path),
        "repaired_library": str(repaired_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
