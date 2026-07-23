from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from autovs.af3 import ToolPending, predict_structure
from autovs.config import Settings
from autovs.dag import (
    AF3_REPORT, AF3_STATE, GROMACS_REPORT, GROMACS_STATE,
    TARGET_STRUCTURE, _bind_gromacs_md, _bind_target_structure_prediction,
)
from autovs.gromacs import submit_gromacs_md
from autovs.planning.graph_builder import (
    PlannedActionIntent, PlannerConstraints, PlannerDraft, WorkflowGraphBuilder,
)
from autovs.schemas import (
    ActionType, InputManifest, LibraryAsset, PocketSpec, TargetAsset, ToolCapability,
)


def _manifest() -> InputManifest:
    return InputManifest(
        query="a sufficiently long screening query for testing",
        library_asset=LibraryAsset(source="user", path="/tmp/lib.smi", sha256="a" * 64),
        target_asset=TargetAsset(source="research", locked=False),
        expert_pocket=PocketSpec(),
    )


def _caps_available() -> list[ToolCapability]:
    return [
        ToolCapability(
            action_type=action,
            name=action.value,
            description="test",
            availability="available",
            executor="python",
            input_formats=[],
            output_formats=[],
            gpu_required=action in {
                ActionType.TARGET_STRUCTURE_PREDICTION,
                ActionType.SHORT_MD,
                ActionType.MOLECULAR_DYNAMICS,
            },
        )
        for action in ActionType
    ]


def test_planner_accepts_af3_and_gromacs_runtime_bindings():
    manifest = _manifest()
    draft = PlannerDraft(
        strategy_id="af3",
        actions=[
            PlannedActionIntent(action_type=ActionType.TARGET_STRUCTURE_PREDICTION, importance="required"),
            PlannedActionIntent(action_type=ActionType.POCKET_DEFINITION, importance="required"),
            PlannedActionIntent(action_type=ActionType.PROTEIN_PREPARATION, importance="required"),
        ],
    )

    result = WorkflowGraphBuilder(
        draft=draft,
        input_manifest=manifest,
        capabilities=_caps_available(),
        constraints=PlannerConstraints(),
    ).build()

    actions = {step.action_type for step in result.plan.steps}
    assert ActionType.TARGET_STRUCTURE_PREDICTION in actions


def test_af3_prediction_downloads_and_binds_structure(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AF3_SERVER_URL", "http://af3.test")
    monkeypatch.setenv("AF3_TOKEN", "secret")
    research = tmp_path / "research.json"
    research.write_text(json.dumps({
        "gene_symbol": "TST",
        "protein_sequence": "M" * 20,
    }), encoding="utf-8")

    def fake_request(env, method, path, **kwargs):  # noqa: ARG001
        if path == "/api/jobs" and method == "POST":
            return 201, json.dumps({"job_id": "job1", "status": "queued"}).encode()
        if path == "/api/jobs/job1":
            return 200, json.dumps({
                "job_id": "job1",
                "status": "succeeded",
                "result_available": True,
                "name": "TST",
            }).encode()
        if path == "/api/jobs/job1/result":
            archive = tmp_path / "result.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("model.pdb", "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n")
            return 200, archive.read_bytes()
        raise AssertionError(path)

    monkeypatch.setattr("autovs.af3._request", fake_request)
    outputs = predict_structure(research_path=research, work_dir=tmp_path / "af3")
    assert Path(outputs["target_structure"]).is_file()

    state = {}
    _bind_target_structure_prediction(outputs, state)
    assert state[TARGET_STRUCTURE] == outputs["target_structure"]
    assert state[AF3_STATE] == outputs["af3_state"]
    assert state[AF3_REPORT] == outputs["af3_report"]


def test_af3_prediction_pauses_when_job_is_running(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AF3_SERVER_URL", "http://af3.test")
    monkeypatch.setenv("AF3_TOKEN", "secret")
    research = tmp_path / "research.json"
    research.write_text(json.dumps({"gene_symbol": "TST", "protein_sequence": "M" * 20}), encoding="utf-8")

    def fake_request(env, method, path, **kwargs):  # noqa: ARG001
        if path == "/api/jobs" and method == "POST":
            return 201, json.dumps({"job_id": "job1", "status": "queued"}).encode()
        if path == "/api/jobs/job1":
            return 200, json.dumps({"job_id": "job1", "status": "running"}).encode()
        raise AssertionError(path)

    monkeypatch.setattr("autovs.af3._request", fake_request)
    with pytest.raises(ToolPending):
        predict_structure(research_path=research, work_dir=tmp_path / "af3")


def test_gromacs_submission_writes_state_and_binder(monkeypatch, tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n")
    sdf = tmp_path / "poses.sdf"
    pytest.importorskip("rdkit")
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("source_id", "mol1")
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()

    apptainer = tmp_path / "apptainer"
    sbatch = tmp_path / "sbatch"
    image = tmp_path / "gromacs.sif"
    for path in (apptainer, sbatch, image):
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    settings = Settings(raw={
        "service": {"database": str(tmp_path / "db.sqlite"), "task_root": str(tmp_path), "host": "127.0.0.1", "port": 1},
        "executables": {"apptainer": str(apptainer), "sbatch": str(sbatch)},
        "executors": {"gromacs": {"name": "gromacs", "executor": "apptainer", "path": str(image)}},
        "limits": {"short_md_hits": 1},
        "libraries": {},
        "environments": {},
        "containers": {},
        "slurm": {"gpu": {"partition": "gpu", "gres": "gpu:1", "cpus": 1, "memory": "1G"}},
    }, config_path=tmp_path / "tools.toml")

    monkeypatch.setattr("src.tools.molecular_utils.SlurmJobManager.submit", lambda *a, **kw: "12345")
    with pytest.raises(ToolPending) as exc:
        submit_gromacs_md(
            receptor_pdb=receptor,
            selected_poses=sdf,
            work_dir=tmp_path / "md",
            settings=settings,
        )
    assert exc.value.payload["slurm_job_id"] == "12345"
    state = {}
    _bind_gromacs_md({
        "gromacs_state": str(tmp_path / "md" / "gromacs_state.json"),
        "gromacs_report": str(tmp_path / "md" / "gromacs_report.json"),
    }, state)
    assert state[GROMACS_STATE].endswith("gromacs_state.json")
    assert state[GROMACS_REPORT].endswith("gromacs_report.json")
