from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def _iter_input(path: Path) -> Iterator[tuple[str, str, dict]]:
    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles", ".txt"}:
        with path.open(encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle, 1):
                fields = line.strip().split()
                if fields:
                    yield fields[0], fields[1] if len(fields) > 1 else f"row_{index:09d}", {}
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return
            smiles_col = next((c for c in reader.fieldnames if c.lower() in {"smiles", "canonical_smiles"}), None)
            if not smiles_col:
                raise ValueError("CSV requires a smiles column")
            id_col = next((c for c in reader.fieldnames if c.lower() in {"source_id", "mol_id", "id", "name"}), None)
            for index, row in enumerate(reader, 1):
                yield row.get(smiles_col, ""), row.get(id_col, "") if id_col else f"row_{index:09d}", row
    elif suffix == ".sdf":
        for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
            if mol is not None:
                props = mol.GetPropsAsDict()
                source = str(props.get("source_id") or props.get("mol_id") or (mol.GetProp("_Name") if mol.HasProp("_Name") else f"row_{index:09d}"))
                yield Chem.MolToSmiles(Chem.RemoveHs(mol)), source, {k: str(v) for k, v in props.items()}
    else:
        raise ValueError(f"unsupported molecular library format: {suffix}")


def _stable_id(smiles: str) -> str:
    return "mol_" + hashlib.sha256(smiles.encode()).hexdigest()[:12]


def prepare_library(input_path: Path, output_dir: Path, *, max_molecules: int = 1_000_000,
                    mw_range: tuple[float, float] = (150.0, 800.0),
                    logp_range: tuple[float, float] = (-2.0, 8.0)) -> dict[str, Path | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    individuals = output_dir / "molecules"
    individuals.mkdir(exist_ok=True)
    combined = output_dir / "prepared_library.sdf"
    manifest = output_dir / "manifest.csv"
    failed = output_dir / "failed.csv"
    summary = output_dir / "summary.tsv"

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains = FilterCatalog(params)
    writer = Chem.SDWriter(str(combined))
    seen: set[str] = set()
    ok_rows, failed_rows = [], []
    total = 0
    for row_index, (raw_smiles, original_id, metadata) in enumerate(_iter_input(input_path)):
        total += 1
        if total > max_molecules:
            writer.close()
            raise ValueError(f"library exceeds configured limit of {max_molecules:,} molecules")
        try:
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                raise ValueError("invalid SMILES")
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in seen:
                raise ValueError("duplicate canonical SMILES")
            seen.add(canonical)
            mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
            if not mw_range[0] <= mw <= mw_range[1]:
                raise ValueError(f"MW {mw:.2f} outside range")
            if not logp_range[0] <= logp <= logp_range[1]:
                raise ValueError(f"LogP {logp:.2f} outside range")
            if pains.GetFirstMatch(mol) is not None:
                raise ValueError("PAINS alert")

            mol = Chem.AddHs(mol)
            embed = AllChem.ETKDGv3()
            embed.randomSeed = 61453 + row_index
            embed.useSmallRingTorsions = True
            embed.useMacrocycleTorsions = True
            status = AllChem.EmbedMolecule(mol, embed)
            if status != 0:
                embed.useRandomCoords = True
                status = AllChem.EmbedMolecule(mol, embed)
            if status != 0:
                raise ValueError("ETKDGv3 embedding failed")
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=300)
                force_field = "MMFF94s"
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=300)
                force_field = "UFF"
            source_id = _stable_id(canonical)
            mol.SetProp("_Name", source_id)
            mol.SetProp("source_id", source_id)
            mol.SetProp("original_id", original_id)
            mol.SetProp("canonical_smiles", canonical)
            for key, value in metadata.items():
                if key and value is not None and key not in {"source_id", "canonical_smiles"}:
                    mol.SetProp(str(key), str(value))
            writer.write(mol)
            one_path = individuals / f"{source_id}.sdf"
            one_writer = Chem.SDWriter(str(one_path)); one_writer.write(mol); one_writer.close()
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol))
            ok_rows.append({
                "source_id": source_id, "original_id": original_id, "smiles": canonical,
                "mw": round(mw, 4), "logp": round(logp, 4),
                "hbd": Lipinski.NumHDonors(mol), "hba": Lipinski.NumHAcceptors(mol),
                "rotatable_bonds": Lipinski.NumRotatableBonds(mol), "scaffold": scaffold,
                "force_field": force_field, "sdf_path": str(one_path),
            })
        except Exception as exc:
            failed_rows.append({"row": row_index + 1, "original_id": original_id, "smiles": raw_smiles, "reason": str(exc)})
    writer.close()
    if not ok_rows:
        raise ValueError("no valid molecules remain after preparation")
    for path, rows, fields in [
        (manifest, ok_rows, list(ok_rows[0])),
        (failed, failed_rows, ["row", "original_id", "smiles", "reason"]),
    ]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            out = csv.DictWriter(handle, fieldnames=fields); out.writeheader(); out.writerows(rows)
    summary.write_text(
        f"input\t{total}\nprepared\t{len(ok_rows)}\nfailed_or_filtered\t{len(failed_rows)}\nzero_explicit_h\t0\n",
        encoding="utf-8",
    )
    return {"prepared_library": combined, "manifest": manifest, "failed": failed,
            "summary": summary, "prepared_count": len(ok_rows), "failed_count": len(failed_rows)}

