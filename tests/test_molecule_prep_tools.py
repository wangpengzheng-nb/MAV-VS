from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from autovs.config import load_settings
from autovs.molecule_prep import (
    enumerate_ionization, obabel_convert, prepare_ligands_3d,
    prepare_pdbqt, standardize_molecules,
)
from autovs.capabilities import health_report


def test_chembl_standardization_desalts_without_header(tmp_path: Path):
    source = tmp_path / "in.smi"
    source.write_text("mol1\tCC(=O)[O-].[Na+]\n", encoding="utf-8")
    out = tmp_path / "std.smi"
    report = standardize_molecules(source, out)
    text = out.read_text(encoding="utf-8")
    assert report["success"] == 1
    assert report["salt_removed"] == 1
    assert not text.lower().startswith("source_id")
    assert "[Na+]" not in text


def test_dimorphite_preserves_source_id():
    states = enumerate_ionization(["mol1\tCN(C)C"], max_states=4)
    assert states
    assert {item["source_id"] for item in states} == {"mol1"}
    assert all("smiles" in item for item in states)


def test_gypsum_meeko_and_obabel_smoke(tmp_path: Path):
    source = tmp_path / "in.smi"
    source.write_text("mol1\tCCO\n", encoding="utf-8")
    sdf = tmp_path / "ligands.sdf"
    report = prepare_ligands_3d(
        source, sdf, max_variants_per_compound=1, max_conformers=1, num_processes=1,
    )
    assert report["returncode"] == 0
    assert sdf.is_file() and sdf.stat().st_size > 0

    pdbqt = tmp_path / "ligands.pdbqt"
    meeko_report = prepare_pdbqt(sdf, pdbqt)
    assert meeko_report["success"] > 0
    assert pdbqt.is_file() and pdbqt.stat().st_size > 0

    obabel = load_settings().executor_config("obabel")
    converted = tmp_path / "converted.sdf"
    obabel_report = obabel_convert(source, converted, obabel_path=obabel.path if obabel else None)
    assert obabel_report["molecules"] > 0
    assert converted.is_file()


def test_meeko_accepts_minimal_3d_sdf(tmp_path: Path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=13)
    AllChem.UFFOptimizeMolecule(mol)
    sdf = tmp_path / "ethanol.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()
    report = prepare_pdbqt(sdf, tmp_path / "ethanol.pdbqt")
    assert report["success"] == 1


def test_capability_health_uses_real_smoke_for_new_molecule_tools():
    report = health_report(load_settings())
    caps = {item["action_type"]: item for item in report["capabilities"]}
    for action in (
        "molecule_standardization_v2",
        "ionization_enumeration",
        "ligand_3d_enumeration",
        "pdbqt_parameterization",
        "format_conversion",
    ):
        assert caps[action]["availability"] == "available", caps[action].get("reason", "")
