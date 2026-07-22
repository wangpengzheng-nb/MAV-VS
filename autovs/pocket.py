from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from autovs.schemas import (
    PocketCandidate,
    PocketConfidence,
    PocketEvidence,
    PocketQualityGate,
    PocketResolution,
    PocketSource,
)
from autovs.security import run_argv


WATER_IONS_ADDITIVES = {
    "HOH", "WAT", "DOD", "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CU", "CO",
    "SO4", "PO4", "NO3", "NH4", "ACT", "ACE", "EDO", "GOL", "PEG", "PG4", "PGE", "MPD",
    "DMS", "BME", "TRS", "MES", "HEP", "MOPS", "CIT", "FMT", "EOH", "IPA", "ACY", "NAG",
}
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
}
INTERACTION_TAGS = {
    "hydrophobic_interaction", "hydrogen_bond", "water_bridge", "salt_bridge", "pi_stack",
    "pi_cation_interaction", "halogen_bond", "metal_complex",
}
RESIDUE_RE = re.compile(r"\b([A-Z]{3})\s*[: -]?\s*(-?\d+)(?:\s*[: -]?\s*([A-Z0-9]))?\b", re.I)


@dataclass(frozen=True)
class Atom:
    record: str
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_number: int
    xyz: tuple[float, float, float]
    element: str

    @property
    def residue_key(self) -> str:
        return f"{self.residue_name}{self.residue_number}{self.chain_id}".upper()


def _parse_pdb(path: Path) -> tuple[list[Atom], list[Atom], set[str], str | None]:
    protein: list[Atom] = []
    hetero: list[Atom] = []
    covalent_residues: set[str] = set()
    pdb_id: str | None = None
    lines = path.read_text(errors="ignore").splitlines()
    if sum(line.startswith("MODEL ") for line in lines) > 1:
        raise ValueError("protein PDB contains multiple MODEL records; upload one prepared receptor conformation")
    for line in lines:
        if line.startswith("HEADER") and len(line) >= 66:
            candidate = line[62:66].strip().upper()
            if re.fullmatch(r"[0-9][A-Z0-9]{3}", candidate):
                pdb_id = candidate
        if line.startswith("LINK") and len(line) >= 57:
            left = f"{line[17:20].strip()}{line[22:26].strip()}{line[21:22].strip()}".upper()
            right = f"{line[47:50].strip()}{line[52:56].strip()}{line[51:52].strip()}".upper()
            left_name, right_name = line[17:20].strip().upper(), line[47:50].strip().upper()
            if left_name in STANDARD_AA and right_name not in STANDARD_AA:
                covalent_residues.add(right)
            elif right_name in STANDARD_AA and left_name not in STANDARD_AA:
                covalent_residues.add(left)
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        if len(line) > 16 and line[16:17] not in {" ", "A"}:
            continue
        try:
            atom = Atom(
                record=line[:6].strip(), serial=int(line[6:11]), name=line[12:16].strip(),
                residue_name=line[17:20].strip().upper(), chain_id=line[21:22].strip().upper(),
                residue_number=int(line[22:26]),
                xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
                element=(line[76:78].strip().upper() if len(line) >= 78 else re.sub(r"[^A-Za-z]", "", line[12:16])[:1].upper()),
            )
        except ValueError:
            continue
        (protein if atom.record == "ATOM" else hetero).append(atom)
    if not protein:
        raise ValueError("protein PDB contains no parseable ATOM records")
    return protein, hetero, covalent_residues, pdb_id


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _box_from_points(points: list[tuple[float, float, float]], *, padding: float = 6.0,
                     minimum: float = 18.0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        raise ValueError("cannot build a pocket box from zero points")
    lows = [min(point[index] for point in points) for index in range(3)]
    highs = [max(point[index] for point in points) for index in range(3)]
    center = tuple(round((lows[index] + highs[index]) / 2.0, 3) for index in range(3))
    size = tuple(round(max(minimum, highs[index] - lows[index] + 2 * padding), 3) for index in range(3))
    return center, size


def _validate_user_box(protein: list[Atom], center: tuple[float, float, float],
                       size: tuple[float, float, float]) -> list[PocketQualityGate]:
    coordinates = [atom.xyz for atom in protein]
    lows = [min(point[index] for point in coordinates) for index in range(3)]
    highs = [max(point[index] for point in coordinates) for index in range(3)]
    inside_expanded_bounds = all(lows[index] - 8.0 <= center[index] <= highs[index] + 8.0 for index in range(3))
    atoms_in_box = sum(
        all(abs(atom.xyz[index] - center[index]) <= size[index] / 2.0 for index in range(3))
        for atom in protein
    )
    gates = [
        PocketQualityGate(name="center_near_protein", status="passed" if inside_expanded_bounds else "failed",
                          detail="center lies within the protein bounds expanded by 8 Angstrom" if inside_expanded_bounds else "center is outside the protein bounds"),
        PocketQualityGate(name="protein_atoms_in_box", status="passed" if atoms_in_box else "failed",
                          detail=f"{atoms_in_box} protein atoms fall inside the docking box"),
    ]
    if not inside_expanded_bounds or not atoms_in_box:
        raise ValueError("provided pocket coordinates do not define a box intersecting the uploaded protein")
    return gates


def _candidate_id(source: PocketSource, center: Iterable[float], residues: Iterable[str]) -> str:
    raw = json.dumps({"source": source.value, "center": list(center), "residues": sorted(residues)}, sort_keys=True)
    return f"pocket_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _candidate(*, rank: int, center: tuple[float, float, float], size: tuple[float, float, float],
               source: PocketSource, confidence: PocketConfidence, chains: list[str], residues: list[str],
               evidence: list[PocketEvidence], gates: list[PocketQualityGate], tools: dict[str, str]) -> PocketCandidate:
    return PocketCandidate(
        pocket_id=_candidate_id(source, center, residues), rank=rank, center=center, size=size,
        source=source, confidence=confidence, chain_ids=sorted(set(chains)), residues=sorted(set(residues)),
        evidence=evidence, quality_gates=gates, tool_versions=tools,
    )


def _selector_matches(selector: str, residue_name: str, chain_id: str, residue_number: int) -> bool:
    compact = re.sub(r"\s+", "", selector).upper()
    choices = {
        residue_name.upper(), f"{residue_name}{residue_number}".upper(),
        f"{residue_name}{residue_number}{chain_id}".upper(),
        f"{residue_name}:{chain_id}:{residue_number}".upper(),
    }
    return compact in choices


def _research_ligand_ids(research: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for structure in research.get("verified_pdb_structures", []) or []:
        if isinstance(structure, dict):
            result.update(str(item).upper() for item in structure.get("ligand_ids", []) if item)
    return result


def _normalize_residue_hint(value: str) -> str | None:
    match = RESIDUE_RE.search(value.upper())
    if not match:
        return None
    name, number, chain = match.groups()
    return f"{name.upper()}{int(number)}{(chain or '').upper()}"


def extract_research_residues(research: dict[str, Any]) -> list[str]:
    """Extract residue hints only from fields explicitly labelled as residue evidence."""
    values: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        normalized_key = key.lower()
        allowed = "residue" in normalized_key or normalized_key in {"binding_site", "structures"}
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            if value.lstrip().startswith("{"):
                try:
                    visit(json.loads(value), key)
                    return
                except json.JSONDecodeError:
                    pass
            if allowed:
                values.extend(match.group(0) for match in RESIDUE_RE.finditer(value.upper()))

    for root_key in ("binding_site", "key_metrics", "executive_summary"):
        if root_key in research:
            visit(research[root_key], root_key)
    normalized = [_normalize_residue_hint(value) for value in values]
    return list(dict.fromkeys(value for value in normalized if value))


def _plip_interaction_counts(protein_path: Path, work_dir: Path, plip_path: Path | None) -> tuple[dict[str, int], str]:
    if not plip_path or not plip_path.exists():
        return {}, "PLIP unavailable"
    output_dir = work_dir / "plip_cocrystal"
    output_dir.mkdir(parents=True, exist_ok=True)
    # 🆕 缓存检查：如果 PLIP 已对此 PDB 运行过，直接复用报告
    existing = [output_dir / "report.xml", *sorted(output_dir.glob("*_report.xml"))]
    cached = next((p for p in existing if p.is_file()), None)
    if cached is not None:
        protonated = next(output_dir.glob("*_protonated.pdb"), None)
        if protonated and protonated.stat().st_mtime > protein_path.stat().st_mtime:
            return _parse_plip_report(cached), "PLIP completed (cached)"
    log_path = work_dir / "plip_cocrystal.log"
    result = run_argv([str(plip_path), "-f", str(protein_path), "-o", str(output_dir), "-x", "-t", "--maxthreads", "1"],
                      cwd=work_dir, timeout=1800, log_path=log_path)
    reports = [output_dir / "report.xml", *sorted(output_dir.glob("*_report.xml"))]
    report = next((path for path in reports if path.is_file()), None)
    if result.returncode or report is None:
        return {}, f"PLIP failed: {result.stderr[-300:] or 'report.xml missing'}"
    return _parse_plip_report(report), "PLIP completed"


def _parse_plip_report(report: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    root = ET.parse(report).getroot()
    for site in root.findall(".//bindingsite"):
        identifiers = site.find("identifiers")
        if identifiers is None:
            continue
        fields = {node.tag.split("}")[-1].lower(): (node.text or "").strip() for node in identifiers}
        name = fields.get("hetid", "").upper()
        chain = fields.get("chain", "").upper()
        position = fields.get("position", "")
        interaction_count = sum(1 for node in site.iter() if node.tag.split("}")[-1].lower() in INTERACTION_TAGS)
        for key in (name, f"{name}{position}", f"{name}{position}{chain}"):
            if key:
                counts[key] = max(counts.get(key, 0), interaction_count)
    return counts


def _ligand_candidates(protein_path: Path, protein: list[Atom], hetero: list[Atom], covalent: set[str],
                       research: dict[str, Any], selector: str | None, work_dir: Path,
                       plip_path: Path | None) -> tuple[list[PocketCandidate], list[str]]:
    grouped: dict[tuple[str, str, int], list[Atom]] = {}
    for atom in hetero:
        grouped.setdefault((atom.residue_name, atom.chain_id, atom.residue_number), []).append(atom)
    research_ligands = _research_ligand_ids(research)
    plip_counts, plip_detail = _plip_interaction_counts(protein_path, work_dir, plip_path)
    warnings = [] if plip_counts else [plip_detail]
    candidates: list[tuple[tuple[int, int, int], PocketCandidate]] = []
    for (name, chain, number), atoms in grouped.items():
        heavy = [atom for atom in atoms if atom.element not in {"H", "D"}]
        residue_key = f"{name}{number}{chain}".upper()
        if name in WATER_IONS_ADDITIVES or name in STANDARD_AA or residue_key in covalent:
            continue
        if selector and not _selector_matches(selector, name, chain, number):
            continue
        if len(heavy) < 6 or not any(atom.element == "C" for atom in heavy):
            continue
        contact_residues: set[str] = set()
        contact_chains: set[str] = set()
        contact_atoms = 0
        for protein_atom in protein:
            if any(_distance(protein_atom.xyz, ligand_atom.xyz) <= 4.5 for ligand_atom in heavy):
                contact_atoms += 1
                contact_residues.add(protein_atom.residue_key)
                contact_chains.add(protein_atom.chain_id)
        if contact_atoms < 3:
            continue
        center, size = _box_from_points([atom.xyz for atom in heavy])
        oversized = any(axis > 36.0 for axis in size)
        plip_count = max(plip_counts.get(name, 0), plip_counts.get(f"{name}{number}", 0), plip_counts.get(residue_key, 0))
        research_match = name in research_ligands
        confidence = PocketConfidence.LOW if oversized else (
            PocketConfidence.HIGH if plip_count > 0 and (research_match or selector) else PocketConfidence.MEDIUM
        )
        gates = [
            PocketQualityGate(name="protein_contacts", status="passed", detail=f"{contact_atoms} protein atoms within 4.5 Angstrom"),
            PocketQualityGate(name="noncovalent", status="passed", detail="no protein-ligand LINK record detected"),
            PocketQualityGate(name="box_extent", status="failed" if oversized else "passed",
                              detail=f"box={size}; maximum allowed automatic dimension is 36 Angstrom"),
            PocketQualityGate(name="plip_interactions", status="passed" if plip_count else ("degraded" if plip_detail == "PLIP completed" else "not_run"),
                              detail=f"{plip_count} PLIP interactions" if plip_count else plip_detail),
        ]
        evidence = [
            PocketEvidence(kind="ligand_identity", description=f"uploaded PDB ligand {name}:{chain}:{number}", value=name),
            PocketEvidence(kind="heavy_atoms", description="ligand heavy atom count", value=len(heavy)),
            PocketEvidence(kind="protein_contacts", description="protein atoms within 4.5 Angstrom", value=contact_atoms),
            PocketEvidence(kind="research_ligand_match", description="ligand CCD appears in verified target research", value=research_match),
        ]
        candidate = _candidate(rank=1, center=center, size=size, source=PocketSource.COCRYSTAL_LIGAND,
                               confidence=confidence, chains=list(contact_chains), residues=list(contact_residues), evidence=evidence,
                               gates=gates, tools={"pdb_parser": "autovs-1", "plip": "configured" if plip_path else "unavailable"})
        priority = (1 if oversized else 0, 0 if research_match or selector else 1, -(plip_count * 1000 + contact_atoms))
        candidates.append((priority, candidate))
    candidates.sort(key=lambda item: item[0])
    return [candidate.model_copy(update={"rank": index}) for index, (_, candidate) in enumerate(candidates, 1)], warnings


def _verified_research_candidate(research: dict[str, Any], uploaded_pdb_id: str | None,
                                 protein: list[Atom], size: tuple[float, float, float]) -> PocketCandidate | None:
    research_pdb = str(research.get("recommended_pdb_for_docking") or "").upper()
    center = research.get("docking_center_from_pdb")
    api_sources = {str(item) for item in research.get("api_sources", []) or []}
    verified_source = f"PDB_ligand_center:{research_pdb}" in api_sources
    if not uploaded_pdb_id or research_pdb != uploaded_pdb_id or not verified_source:
        return None
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return None
    try:
        center_tuple = tuple(round(float(item), 3) for item in center)
    except (TypeError, ValueError):
        return None
    try:
        gates = _validate_user_box(protein, center_tuple, size)
    except ValueError:
        return None
    gates.append(PocketQualityGate(name="same_pdb_coordinate_frame", status="passed",
                                   detail=f"uploaded HEADER and verified research both identify {research_pdb}"))
    return _candidate(
        rank=1, center=center_tuple, size=size, source=PocketSource.VERIFIED_RESEARCH_STRUCTURE,
        confidence=PocketConfidence.HIGH, chains=[], residues=[],
        evidence=[PocketEvidence(kind="verified_api_coordinate", description="ligand center computed from the verified RCSB PDB", value=research_pdb)],
        gates=gates, tools={"target_scout": "verified-api-output"},
    )


def _residue_candidate(protein: list[Atom], hints: list[str], *, explicit: bool) -> PocketCandidate | None:
    normalized = [_normalize_residue_hint(item) for item in hints]
    requested = [item for item in normalized if item]
    if not requested:
        return None
    matched_atoms: list[Atom] = []
    matched_keys: set[str] = set()
    for atom in protein:
        no_chain = f"{atom.residue_name}{atom.residue_number}".upper()
        with_chain = f"{no_chain}{atom.chain_id}".upper()
        if any(hint in {no_chain, with_chain} for hint in requested):
            matched_atoms.append(atom)
            matched_keys.add(with_chain)
    minimum = 1 if explicit else 2
    if len(matched_keys) < minimum:
        return None
    center, size = _box_from_points([atom.xyz for atom in matched_atoms])
    oversized = any(axis > 36.0 for axis in size)
    match_ratio = len(matched_keys) / max(1, len(requested))
    confidence = PocketConfidence.LOW if oversized else (
        PocketConfidence.HIGH if explicit and match_ratio >= 1.0 else PocketConfidence.MEDIUM
    )
    gates = [
        PocketQualityGate(name="residue_mapping", status="passed" if match_ratio >= 0.5 else "degraded",
                          detail=f"matched {len(matched_keys)} of {len(requested)} residue hints"),
        PocketQualityGate(name="box_extent", status="failed" if oversized else "passed",
                          detail=f"box={size}; maximum allowed automatic dimension is 36 Angstrom"),
    ]
    return _candidate(
        rank=1, center=center, size=size, source=PocketSource.KEY_RESIDUES, confidence=confidence,
        chains=[atom.chain_id for atom in matched_atoms], residues=list(matched_keys),
        evidence=[PocketEvidence(kind="residue_hints", description="residues mapped onto the uploaded PDB", value=";".join(sorted(matched_keys)))],
        gates=gates, tools={"pdb_parser": "autovs-1"},
    )


def resolve_pocket(protein_path: Path, *, center: tuple[float, float, float] | None,
                   size: tuple[float, float, float], key_residues: list[str],
                   cocrystal_ligand: str | None = None, research: dict[str, Any] | None = None,
                   work_dir: Path | None = None, plip_path: Path | None = None) -> PocketResolution:
    """Resolve a traceable docking pocket in the uploaded PDB coordinate frame.

    LLM text never supplies coordinates. Research coordinates are accepted only when TargetScout
    marked them as verified API output and the uploaded PDB HEADER identifies the same PDB entry.
    """
    research = research or {}
    work_dir = work_dir or protein_path.parent
    protein, hetero, covalent, uploaded_pdb_id = _parse_pdb(protein_path)
    if center is not None:
        gates = _validate_user_box(protein, center, size)
        selected = _candidate(
            rank=1, center=tuple(round(float(item), 3) for item in center), size=size,
            source=PocketSource.USER_COORDINATES, confidence=PocketConfidence.HIGH,
            chains=[], residues=[],
            evidence=[PocketEvidence(kind="user_input", description="pocket center supplied explicitly by the user", value=True)],
            gates=gates, tools={"pdb_parser": "autovs-1"},
        )
        return PocketResolution(protein_path=str(protein_path), selected_pocket=selected,
                                research_pdb_id=str(research.get("recommended_pdb_for_docking") or "") or None)

    ligand_candidates, warnings = _ligand_candidates(
        protein_path, protein, hetero, covalent, research, cocrystal_ligand, work_dir, plip_path,
    )
    if cocrystal_ligand and not ligand_candidates:
        raise ValueError(f"specified cocrystal ligand {cocrystal_ligand!r} was not found or failed pocket quality gates")
    research_candidate = _verified_research_candidate(research, uploaded_pdb_id, protein, size)
    hints = key_residues or extract_research_residues(research)
    residue_candidate = _residue_candidate(protein, hints, explicit=bool(key_residues))
    usable_ligands = [item for item in ligand_candidates if item.confidence != PocketConfidence.LOW]
    ordered: list[PocketCandidate] = []
    if key_residues and residue_candidate and residue_candidate.confidence != PocketConfidence.LOW:
        ordered.append(residue_candidate)
    ordered.extend(item for item in usable_ligands if all(_distance(item.center, chosen.center) > 2.0 for chosen in ordered))
    if research_candidate and all(_distance(research_candidate.center, item.center) > 2.0 for item in ordered):
        ordered.append(research_candidate)
    if residue_candidate and residue_candidate.confidence != PocketConfidence.LOW and all(
        _distance(residue_candidate.center, item.center) > 4.0 for item in ordered
    ):
        ordered.append(residue_candidate)
    if not ordered:
        reasons = ["no user coordinates", "no usable noncovalent ligand in uploaded PDB"]
        if not hints:
            reasons.append("research supplied no mappable key residues")
        if uploaded_pdb_id != str(research.get("recommended_pdb_for_docking") or "").upper():
            reasons.append("verified research coordinates belong to a different PDB coordinate frame")
        raise ValueError("pocket unresolved: " + "; ".join(reasons))
    ranked = [item.model_copy(update={"rank": index}) for index, item in enumerate(ordered[:3], 1)]
    return PocketResolution(
        protein_path=str(protein_path), selected_pocket=ranked[0], alternate_pockets=ranked[1:3],
        research_pdb_id=str(research.get("recommended_pdb_for_docking") or "") or None,
        warnings=warnings,
    )
