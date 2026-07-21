#!/usr/bin/env python3
"""Rebuild the versioned strict-SMI default library from merged_dedup.smi."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase


EXPECTED_SOURCE_SHA256 = "1bac7c742cc795712a5c52f7df440bfac03c2c7728dcda7e80455cd62ea7f7a6"
EXPECTED_NORMALIZED_SHA256 = "c6d2c6aec202f07b9abf8bed30e4b31af756999f05702dedd1a9a02882af2353"
VERSION = "pocketxmol_87924_v1"
RDLogger.DisableLog("rdApp.*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/libraries"))
    args = parser.parse_args()
    source = args.source.resolve()
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("source checksum does not match the audited merged_dedup.smi")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{VERSION}.smi"
    rejected_path = args.output_dir / f"{VERSION}.rejected.tsv"
    output_tmp = args.output_dir / f".{VERSION}.smi.tmp"
    rejected_tmp = args.output_dir / f".{VERSION}.rejected.tsv.tmp"
    accepted: list[tuple[str, str]] = []
    rejected: list[tuple[int, str, str, str]] = []
    owners: dict[str, str] = {}
    ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            rejected.append((line_number, "", "", "source_format_error")); continue
        smiles, molecule_id = fields
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected.append((line_number, molecule_id, smiles, "invalid_smiles")); continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if molecule_id in ids:
            rejected.append((line_number, molecule_id, smiles, "duplicate_molecule_id")); continue
        if canonical in owners:
            rejected.append((line_number, molecule_id, smiles, f"duplicate_structure_of:{owners[canonical]}")); continue
        ids.add(molecule_id); owners[canonical] = molecule_id; accepted.append((molecule_id, canonical))
    output_tmp.write_text("".join(f"{molecule_id}\t{smiles}\n" for molecule_id, smiles in accepted), encoding="utf-8")
    rejected_tmp.write_text(
        "line_number\tmolecule_id\tsmiles\treason\n" +
        "".join(f"{line}\t{molecule_id}\t{smiles}\t{reason}\n"
        for line, molecule_id, smiles, reason in rejected), encoding="utf-8",
    )
    output_tmp.replace(output)
    rejected_tmp.replace(rejected_path)
    normalized_sha = sha256(output)
    if len(accepted) != 87924 or len(rejected) != 17 or normalized_sha != EXPECTED_NORMALIZED_SHA256:
        raise SystemExit("rebuilt library does not match the audited v1 output")
    provenance = {
        "library_name": "PocketXMol curated 87K", "version": VERSION, "format": "strict_smi_v1",
        "source_filename": source.name, "source_sha256": sha256(source), "source_records": 87941,
        "conversion": "source SMILES/ID whitespace pairs -> molecule_id<TAB>canonical_isomeric_SMILES; invalid SMILES quarantined",
        "rdkit_version": rdBase.rdkitVersion, "accepted_records": len(accepted),
        "quarantined_records": len(rejected), "normalized_filename": output.name,
        "normalized_sha256": normalized_sha, "rejected_filename": rejected_path.name,
        "rejected_sha256": sha256(rejected_path),
    }
    provenance_path = args.output_dir / f"{VERSION}.provenance.json"
    provenance_tmp = args.output_dir / f".{VERSION}.provenance.json.tmp"
    provenance_tmp.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    provenance_tmp.replace(provenance_path)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
