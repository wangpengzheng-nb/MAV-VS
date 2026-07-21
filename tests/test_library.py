import json

import pytest

from autovs.config import load_settings
from autovs.library import (SmiFormatError, migrate_legacy_library, normalize_smi_library,
                            validate_smi_structure, verify_default_library)


@pytest.mark.parametrize("content,error_type", [
    ("mol1 CCO\n", "column_count"),
    ("molecule_id\tsmiles\n", "header_forbidden"),
    ("mol1\tCCO\textra\n", "column_count"),
    ("# comment\n", "comment_line"),
    ("mol1\tCCO\n\nmol2\tCCN\n", "blank_line"),
    ("bad/id\tCCO\n", "invalid_molecule_id"),
])
def test_strict_smi_rejects_structural_format_errors(tmp_path, content, error_type):
    path = tmp_path / "library.smi"; path.write_text(content)
    with pytest.raises(SmiFormatError) as caught:
        validate_smi_structure(path)
    assert caught.value.error_type == error_type


def test_strict_smi_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "library.smi"; path.write_bytes(b"mol1\tCCO\n\xff")
    with pytest.raises(SmiFormatError, match="UTF-8"):
        validate_smi_structure(path)


def test_chemical_errors_are_quarantined_and_ids_are_preserved(tmp_path):
    source = tmp_path / "library.smi"
    source.write_text("ethanol\tCCO\nbad\tnot-smiles\nethanol\tCCN\nsame\tOCC\nwater\tO\n")
    result = normalize_smi_library(source, tmp_path / "out")
    assert result["total_records"] == 5
    assert result["accepted_records"] == 2
    assert result["quarantined_records"] == 3
    assert result["normalized_library"].read_text().splitlines()[0] == "ethanol\tCCO"
    rejected = result["rejected"].read_text()
    assert "invalid_smiles" in rejected
    assert "duplicate_molecule_id" in rejected
    assert "duplicate_structure_of:ethanol" in rejected
    payload = json.loads(result["validation"].read_text())
    assert payload["format"] == "strict_smi_v1"


def test_zero_valid_molecules_fails(tmp_path):
    source = tmp_path / "library.smi"; source.write_text("bad\tnot-smiles\n")
    with pytest.raises(ValueError, match="no valid molecules"):
        normalize_smi_library(source, tmp_path / "out")


def test_project_default_library_is_versioned_and_checksummed():
    settings = load_settings()
    cfg = settings.library()
    result = verify_default_library(settings.default_library_path, cfg["sha256"], cfg["molecule_count"])
    assert result["status"] == "available"
    assert result["molecule_count"] == 87924
    assert result["sha256"] == "c6d2c6aec202f07b9abf8bed30e4b31af756999f05702dedd1a9a02882af2353"


def test_persisted_legacy_smiles_first_library_can_be_migrated(tmp_path):
    legacy = tmp_path / "old.smi"; legacy.write_text("CCO ethanol\nCCN ethylamine\n")
    migrated = migrate_legacy_library(legacy, tmp_path / "legacy_migrated.smi")
    assert migrated.read_text() == "ethanol\tCCO\nethylamine\tCCN\n"
    assert validate_smi_structure(migrated) == 2
