from __future__ import annotations

from pathlib import Path
import json

import pytest

from autovs.pocket import extract_research_residues, resolve_pocket
from autovs.config import Settings
from autovs.db import StateStore
from autovs.schemas import ActionType, JobStatus, PocketConfidence, PocketSource, WorkflowStep
from autovs.tool_manager import ToolManager


def atom_line(serial: int, name: str, residue: str, chain: str, number: int,
              x: float, y: float, z: float, *, hetero: bool = False, element: str = "C") -> str:
    record = "HETATM" if hetero else "ATOM  "
    return (
        f"{record}{serial:5d} {name:^4s} {residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2s}  "
    )


def write_pdb(path: Path, *, ligand: bool = False, pdb_id: str = "1ABC") -> Path:
    lines = [f"HEADER    TEST PROTEIN                            01-JAN-00   {pdb_id}"]
    serial = 1
    for residue, number, base in (("ASP", 10, 0.0), ("TRP", 20, 4.0), ("GLY", 30, 8.0)):
        for atom_name, dx, element in (("N", 0.0, "N"), ("CA", 0.8, "C"), ("C", 1.5, "C"), ("O", 2.0, "O")):
            lines.append(atom_line(serial, atom_name, residue, "A", number, base + dx, 0.0, 0.0, element=element))
            serial += 1
    lines.append(atom_line(serial, "O", "HOH", "W", 1, 2.0, 8.0, 8.0, hetero=True, element="O")); serial += 1
    if ligand:
        for index in range(6):
            lines.append(atom_line(serial, f"C{index + 1}", "LIG", "L", 101,
                                   2.0 + index * 0.4, 1.5, 0.0, hetero=True, element="C"))
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_user_coordinates_are_validated_and_selected(tmp_path):
    pdb = write_pdb(tmp_path / "protein.pdb")
    result = resolve_pocket(pdb, center=(4.0, 0.0, 0.0), size=(20.0, 20.0, 20.0),
                            key_residues=[], work_dir=tmp_path)
    assert result.selected_pocket.source == PocketSource.USER_COORDINATES
    assert result.selected_pocket.confidence == PocketConfidence.HIGH
    with pytest.raises(ValueError, match="do not define"):
        resolve_pocket(pdb, center=(200.0, 200.0, 200.0), size=(20.0, 20.0, 20.0),
                       key_residues=[], work_dir=tmp_path)


def test_uploaded_cocrystal_ligand_defines_box_and_ignores_water(tmp_path):
    pdb = write_pdb(tmp_path / "holo.pdb", ligand=True)
    result = resolve_pocket(pdb, center=None, size=(24.0, 24.0, 24.0), key_residues=[],
                            cocrystal_ligand="LIG:L:101", research={}, work_dir=tmp_path)
    pocket = result.selected_pocket
    assert pocket.source == PocketSource.COCRYSTAL_LIGAND
    assert pocket.center[0] == pytest.approx(3.0)
    assert pocket.size == (18.0, 18.0, 18.0)
    assert any(item.kind == "ligand_identity" and "LIG" in item.description for item in pocket.evidence)


def test_verified_research_coordinate_requires_same_uploaded_pdb(tmp_path):
    pdb = write_pdb(tmp_path / "protein.pdb", pdb_id="1ABC")
    research = {
        "recommended_pdb_for_docking": "1ABC", "docking_center_from_pdb": [4.0, 0.0, 0.0],
        "api_sources": ["PDB_ligand_center:1ABC"],
    }
    result = resolve_pocket(pdb, center=None, size=(24.0, 24.0, 24.0), key_residues=[],
                            research=research, work_dir=tmp_path)
    assert result.selected_pocket.source == PocketSource.VERIFIED_RESEARCH_STRUCTURE
    research["recommended_pdb_for_docking"] = "2XYZ"
    research["api_sources"] = ["PDB_ligand_center:2XYZ"]
    with pytest.raises(ValueError, match="different PDB coordinate frame"):
        resolve_pocket(pdb, center=None, size=(24.0, 24.0, 24.0), key_residues=[],
                       research=research, work_dir=tmp_path)


def test_research_residues_are_mapped_in_uploaded_coordinate_frame(tmp_path):
    pdb = write_pdb(tmp_path / "apo.pdb")
    research = {"binding_site": {"key_residues_text": "ASP10 and TRP20 form the binding site"}}
    result = resolve_pocket(pdb, center=None, size=(24.0, 24.0, 24.0), key_residues=[],
                            research=research, work_dir=tmp_path)
    assert result.selected_pocket.source == PocketSource.KEY_RESIDUES
    assert {"ASP10A", "TRP20A"}.issubset(result.selected_pocket.residues)


def test_json_executive_summary_residue_extraction():
    research = {"executive_summary": '{"structures":[{"key_residues":["ASP103","TRP144A"]}]}' }
    assert extract_research_residues(research) == ["ASP103", "TRP144A"]


def test_explicit_residues_override_unselected_uploaded_ligand(tmp_path):
    pdb = write_pdb(tmp_path / "holo.pdb", ligand=True)
    result = resolve_pocket(pdb, center=None, size=(24.0, 24.0, 24.0),
                            key_residues=["TRP20", "GLY30"], research={}, work_dir=tmp_path)
    assert result.selected_pocket.source == PocketSource.KEY_RESIDUES
    assert result.alternate_pockets[0].source == PocketSource.COCRYSTAL_LIGAND


def test_missing_explicit_ligand_selector_fails_instead_of_silent_fallback(tmp_path):
    pdb = write_pdb(tmp_path / "holo.pdb", ligand=True)
    with pytest.raises(ValueError, match="specified cocrystal ligand"):
        resolve_pocket(pdb, center=None, size=(24.0, 24.0, 24.0), key_residues=["ASP10", "TRP20"],
                       cocrystal_ligand="NOPE:Z:999", research={}, work_dir=tmp_path)


def test_multiple_model_pdb_is_rejected(tmp_path):
    pdb = write_pdb(tmp_path / "multi.pdb")
    text = pdb.read_text()
    pdb.write_text("MODEL        1\n" + text + "MODEL        2\n" + text)
    with pytest.raises(ValueError, match="multiple MODEL"):
        resolve_pocket(pdb, center=(4.0, 0.0, 0.0), size=(20.0, 20.0, 20.0),
                       key_residues=[], work_dir=tmp_path)


def test_tool_manager_accepts_task_research_artifact_and_rejects_inline_research(tmp_path):
    task_root = tmp_path / "tasks"; task_dir = task_root / "task"; task_dir.mkdir(parents=True)
    pdb = write_pdb(task_dir / "protein.pdb")
    research_path = task_dir / "research.json"
    research_path.write_text(json.dumps({"binding_site": {"key_residues_text": "ASP10 and TRP20"}}))
    settings = Settings(raw={
        "service": {"database": str(tmp_path / "state.sqlite3"), "task_root": str(task_root), "host": "127.0.0.1", "port": 8765},
        "executables": {"plip": ""}, "limits": {}, "environments": {}, "containers": {},
    }, config_path=tmp_path / "tools.toml")
    store = StateStore(settings.database_path)
    task_id = store.create_task({"query": "a sufficiently long pocket test"}, task_dir)
    manager = ToolManager(settings, store)
    job = manager.submit(task_id, WorkflowStep(step_id="pocket", action_type=ActionType.POCKET_DEFINITION), {
        "protein_path": str(pdb), "center": None, "size": (24, 24, 24), "key_residues": [],
        "research_path": str(research_path),
    }, background=False)
    assert job.status == JobStatus.SUCCEEDED
    resolution = json.loads(Path(json.loads(job.message)["pocket"]).read_text())
    assert resolution["selected_pocket"]["source"] == "key_residues"

    rejected = manager.submit(task_id, WorkflowStep(step_id="pocket-inline", action_type=ActionType.POCKET_DEFINITION), {
        "protein_path": str(pdb), "center": (4, 0, 0), "size": (24, 24, 24), "key_residues": [],
        "research": {"docking_center_from_pdb": [4, 0, 0]},
    }, background=False)
    assert rejected.status == JobStatus.FAILED
    assert "inline research is forbidden" in rejected.message
