from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from autovs.security import sha256_file


PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
RCSB_DOWNLOAD_ORIGIN = "https://files.rcsb.org"
SKIP_HET = {"HOH", "H2O", "DOD", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4", "GOL", "EDO", "ACT", "MPD", "PEG"}


def rank_verified_holo_candidates(research: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    target_uniprot = str(research.get("target_uniprot_id") or research.get("uniprot_id") or "")
    target_organism = str(research.get("target_organism") or "")
    if not target_uniprot or target_uniprot == "N/A" or not target_organism:
        raise ValueError("research could not establish a verified UniProt and organism identity for structure acquisition")
    candidates = []
    for raw in research.get("verified_pdb_structures", []):
        item = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        pdb_id = str(item.get("pdb_id", "")).upper()
        if (not PDB_ID_RE.fullmatch(pdb_id) or not item.get("has_ligand")
                or not item.get("uniprot_mapped")):
            continue
        item["pdb_id"] = pdb_id
        item["target_uniprot_id"] = target_uniprot
        item["target_organism"] = target_organism
        candidates.append(item)
    candidates.sort(key=lambda item: (
        float(item.get("resolution") or 99.0),
        -int(item.get("deposition_year") or 0),
        item["pdb_id"],
    ))
    return candidates[:max(1, min(limit, 5))]


def acquire_rcsb_structures(research_path: Path, output_dir: Path, *, limit: int = 5,
                            selected_strategy_id: str = "") -> dict[str, object]:
    research = json.loads(research_path.read_text(encoding="utf-8"))
    if not isinstance(research, dict):
        raise ValueError("research artifact must contain a JSON object")
    candidates = rank_verified_holo_candidates(research, limit=limit)
    if not candidates:
        raise ValueError("research found no verified experimental holo PDB candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    headers = {"User-Agent": "AutoVS-Agent/0.1 (+local-controlled-RCSB-client)"}
    with httpx.Client(timeout=45.0, follow_redirects=False, headers=headers) as client:
        for candidate in candidates:
            pdb_id = str(candidate["pdb_id"])
            url = f"{RCSB_DOWNLOAD_ORIGIN}/download/{pdb_id}.pdb"
            try:
                response = client.get(url)
                response.raise_for_status()
                text = response.text
                if not any(line.startswith("ATOM  ") for line in text.splitlines()):
                    raise ValueError("downloaded file contains no ATOM records")
                ligand_ids = _non_solvent_hetero_ids(text)
                if not ligand_ids:
                    raise ValueError("downloaded structure contains no non-solvent ligand")
                path = output_dir / f"{pdb_id}.pdb"
                path.write_text(text, encoding="utf-8")
                downloaded.append({
                    "pdb_id": pdb_id, "path": str(path), "sha256": sha256_file(path),
                    "url": url, "resolution": candidate.get("resolution"),
                    "deposition_year": candidate.get("deposition_year"),
                    "ligand_ids": ligand_ids, "target_uniprot_id": candidate.get("target_uniprot_id", ""),
                    "target_organism": candidate.get("target_organism", ""),
                })
            except (httpx.HTTPError, ValueError) as exc:
                rejected.append({"pdb_id": pdb_id, "reason": str(exc)})
    report = output_dir / "structure_acquisition.json"
    payload = {
        "source": "RCSB PDB", "allowed_origin": RCSB_DOWNLOAD_ORIGIN,
        "selected_strategy_id": selected_strategy_id, "candidate_count": len(candidates),
        "downloaded": downloaded, "rejected": rejected,
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not downloaded:
        raise RuntimeError(f"all verified RCSB structure downloads failed; rejected={rejected}")
    return {"candidate_structures": [item["path"] for item in downloaded],
            "candidates": downloaded, "acquisition_report": report}


def _non_solvent_hetero_ids(pdb_text: str) -> list[str]:
    values = set()
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        residue = line[17:20].strip().upper()
        if residue and residue not in SKIP_HET:
            values.add(residue)
    return sorted(values)
