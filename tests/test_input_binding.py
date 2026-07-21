import json

import pytest

from autovs.config import Settings
from autovs.pipeline import PipelineService
from autovs.schemas import TaskRequest
from autovs.security import sha256_file


@pytest.fixture
def binding_service(tmp_path):
    default = tmp_path / "default.smi"; default.write_text("builtin1\tCCO\n")
    raw = {
        "service": {"database": str(tmp_path / "state.sqlite3"), "task_root": str(tmp_path / "tasks"),
                    "host": "127.0.0.1", "port": 8765},
        "limits": {"max_library_molecules": 100}, "executables": {}, "environments": {}, "containers": {},
        "libraries": {"default": {"path": str(default), "version": "test_v1", "format": "strict_smi_v1",
                                  "sha256": sha256_file(default), "molecule_count": 1}},
    }
    return PipelineService(Settings(raw=raw, config_path=tmp_path / "tools.toml"))


@pytest.mark.parametrize("with_library,with_protein,library_source,target_source,target_locked", [
    (False, False, "builtin", "research", False),
    (True, False, "user", "research", False),
    (False, True, "builtin", "user", True),
    (True, True, "user", "user", True),
])
def test_four_input_binding_combinations(binding_service, tmp_path, with_library, with_protein,
                                         library_source, target_source, target_locked):
    library = tmp_path / f"user_{with_library}_{with_protein}.smi"
    library.write_text("user1\tCCN\n")
    protein = tmp_path / f"protein_{with_library}_{with_protein}.pdb"
    protein.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n")
    staged, _ = binding_service._stage_request(TaskRequest(
        query="a sufficiently detailed virtual screening request",
        library_path=str(library) if with_library else None,
        protein_path=str(protein) if with_protein else None,
    ))
    manifest = json.loads(open(staged.input_manifest_path, encoding="utf-8").read())
    assert manifest["library_asset"]["source"] == library_source
    assert manifest["library_asset"]["locked"] is True
    assert manifest["target_asset"]["source"] == target_source
    assert manifest["target_asset"]["locked"] is target_locked
    assert bool(staged.protein_path) is with_protein
    assert staged.library_path
    if not with_library:
        assert any("内置" in warning for warning in manifest["warnings"])
