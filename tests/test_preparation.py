import csv

from rdkit import Chem

from autovs.preparation import prepare_library


def test_prepare_library_is_deterministic_and_explicit_h(tmp_path):
    source = tmp_path / "input.smi"
    source.write_text("aspirin\tCC(=O)Oc1ccccc1C(=O)O\ncaffeine\tCn1c(=O)c2c(ncn2C)n(C)c1=O\nbad\tnot-a-smiles\nduplicate\tCC(=O)Oc1ccccc1C(=O)O\n")
    result = prepare_library(source, tmp_path / "out")
    rows = list(csv.DictReader(open(result["manifest"], encoding="utf-8")))
    failed = list(csv.DictReader(open(result["failed"], encoding="utf-8")))
    assert len(rows) == 2
    assert len(failed) == 2
    molecules = [m for m in Chem.SDMolSupplier(str(result["prepared_library"]), removeHs=False) if m]
    assert all(any(atom.GetSymbol() == "H" for atom in mol.GetAtoms()) for mol in molecules)
    assert rows[0]["source_id"] == "aspirin"
    assert rows[0]["structure_id"].startswith("mol_")
