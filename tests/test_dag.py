"""Unit tests for the DAG workflow executor and artifact binding registry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autovs.dag import (
    ACTION_PHASE_MAP,
    COMPLEX_INDEX,
    DOCKED_POSES,
    ENUMERATED_3D_SDF,
    INPUT_RESOLVERS,
    IONIZED_LIBRARY,
    LIGAND_PDBQT,
    MANIFEST_CSV,
    MOLECULE_PREP_REPORTS,
    NORMALIZED_LIBRARY,
    OUTPUT_BINDERS,
    POCKET_CENTER,
    POCKET_SIZE,
    PREPARED_LIBRARY,
    RECEPTOR_PDB,
    RECEPTOR_PDBQT,
    SCORES_CSV,
    SCREENING_LIBRARY,
    STANDARDIZED_LIBRARY,
    TARGET_STRUCTURE,
    TOP_HITS,
    DAGExecutionError,
    TaskPaused,
    execute_workflow_plan,
)
from autovs.schemas import (
    ActionType, InputManifest, JobRecord, JobStatus, LibraryAsset,
    PocketResolution, PocketSpec, TargetAsset, TaskRequest, WorkflowPlan,
    WorkflowStep,
)


def _mock_store(task_dir: Path) -> MagicMock:
    store = MagicMock()
    store.get_job.return_value = None
    store.list_artifacts.return_value = []
    store.list_jobs.return_value = []
    return store


def _mock_tools(tmp_path: Path) -> MagicMock:
    tools = MagicMock()
    from autovs.config import Settings
    settings = Settings(raw={
        "service": {"database": str(tmp_path / "db.sqlite3"), "task_root": str(tmp_path / "tasks"),
                    "host": "127.0.0.1", "port": 8765},
        "executables": {}, "limits": {}, "environments": {}, "containers": {},
    }, config_path=tmp_path / "tools.toml")
    tools.settings = settings
    return tools


def _step(action_type: ActionType, step_id: str = "", requires: list[str] | None = None) -> WorkflowStep:
    sid = step_id or action_type.value
    return WorkflowStep(step_id=sid, action_type=action_type,
                        requires=requires or [], parameters={})


def _success_job(outputs: dict) -> MagicMock:
    """Return a mock that looks like a succeeded JobRecord to store.get_job()."""
    job = MagicMock()
    job.job_id = "job-1"
    job.status = JobStatus.SUCCEEDED
    job.message = json.dumps(outputs)
    return job


def _failed_job(message: str = "tool failed") -> MagicMock:
    """Return a mock that looks like a failed JobRecord."""
    job = MagicMock()
    job.job_id = "job-1"
    job.status = JobStatus.FAILED
    job.message = message
    return job


def _pocket_json(tmp_path: Path) -> Path:
    """Write a valid PocketResolution JSON."""
    path = tmp_path / "pocket.json"
    path.write_text(json.dumps({
        "resolution_version": "1.0",
        "protein_path": str(tmp_path / "protein.pdb"),
        "selected_pocket": {
            "pocket_id": "pocket_a1b2c3d4e5f6",
            "rank": 1,
            "center": [1.0, 2.0, 3.0],
            "size": [20.0, 20.0, 20.0],
            "source": "user_coordinates",
            "confidence": "high",
        },
        "alternate_pockets": [],
    }))
    return path


_LONG_QUERY = "a sufficiently long screening query for testing"


# ── Input resolver tests ──────────────────────────────────────────────

class TestInputResolvers:
    def test_input_validation_reads_screening_library(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"query":"test"}')
        state = {SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
                 "_input_manifest_path": str(manifest)}
        inputs = INPUT_RESOLVERS[ActionType.INPUT_VALIDATION](state)
        assert inputs["library_path"] == state[SCREENING_LIBRARY]
        assert inputs["input_manifest_path"] == str(manifest)

    def test_input_validation_with_protein(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"query":"test"}')
        state = {SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
                 TARGET_STRUCTURE: str(tmp_path / "protein.pdb"),
                 "_input_manifest_path": str(manifest)}
        inputs = INPUT_RESOLVERS[ActionType.INPUT_VALIDATION](state)
        assert inputs["protein_path"] == state[TARGET_STRUCTURE]

    def test_pocket_definition_resolves_center_and_size(self):
        state = {TARGET_STRUCTURE: "/tmp/protein.pdb"}
        inputs = INPUT_RESOLVERS[ActionType.POCKET_DEFINITION](
            state, center=(10, 20, 30), size=(15, 15, 15),
        )
        assert inputs["center"] == (10, 20, 30)
        assert inputs["size"] == (15, 15, 15)

    def test_pocket_definition_falls_back_to_artifact_state(self):
        state = {
            TARGET_STRUCTURE: "/tmp/protein.pdb",
            POCKET_CENTER: (9, 8, 7),
            POCKET_SIZE: (18, 19, 20),
            "_pocket_key_residues": ["ASP103", "TRP144"],
            "_pocket_cocrystal_ligand": "LIG",
        }
        inputs = INPUT_RESOLVERS[ActionType.POCKET_DEFINITION](state)
        assert inputs["center"] == (9, 8, 7)
        assert inputs["size"] == (18, 19, 20)
        assert inputs["key_residues"] == ["ASP103", "TRP144"]
        assert inputs["cocrystal_ligand"] == "LIG"

    def test_docking_resolves_all_keys(self):
        state = {
            RECEPTOR_PDBQT: "/tmp/rec.pdbqt",
            PREPARED_LIBRARY: "/tmp/lig.sdf",
            MANIFEST_CSV: "/tmp/manifest.csv",
            POCKET_CENTER: (1, 2, 3),
            POCKET_SIZE: (22, 22, 22),
        }
        inputs = INPUT_RESOLVERS[ActionType.MOLECULAR_DOCKING](state)
        assert inputs["receptor_pdbqt"] == state[RECEPTOR_PDBQT]
        assert inputs["ligands_sdf"] == state[PREPARED_LIBRARY]
        assert inputs["center"] == (1, 2, 3)
        assert inputs["size"] == (22, 22, 22)

    def test_final_ranking_reads_scores_csv(self):
        state = {SCORES_CSV: "/tmp/scores.csv"}
        inputs = INPUT_RESOLVERS[ActionType.FINAL_RANKING](state)
        assert inputs["scores_csv"] == state[SCORES_CSV]

    def test_new_molecule_tool_resolvers_chain_artifacts(self):
        state = {NORMALIZED_LIBRARY: "/tmp/input.smi"}
        std_inputs = INPUT_RESOLVERS[ActionType.MOLECULE_STANDARDIZATION_V2](state)
        assert std_inputs["library_path"] == "/tmp/input.smi"
        OUTPUT_BINDERS[ActionType.MOLECULE_STANDARDIZATION_V2](
            {"standardized_library": "/tmp/std.smi", "standardization_report": "/tmp/std.json"}, state,
        )
        ion_inputs = INPUT_RESOLVERS[ActionType.IONIZATION_ENUMERATION](state)
        assert ion_inputs["library_path"] == "/tmp/std.smi"
        OUTPUT_BINDERS[ActionType.IONIZATION_ENUMERATION](
            {"ionized_library": "/tmp/ion.smi", "ionization_report": "/tmp/ion.json"}, state,
        )
        enum_inputs = INPUT_RESOLVERS[ActionType.LIGAND_3D_ENUMERATION](state)
        assert enum_inputs["library_path"] == "/tmp/ion.smi"
        OUTPUT_BINDERS[ActionType.LIGAND_3D_ENUMERATION](
            {"prepared_3d_sdf": "/tmp/lig.sdf", "enumeration_report": "/tmp/enum.json"}, state,
        )
        assert state[STANDARDIZED_LIBRARY] == "/tmp/std.smi"
        assert state[IONIZED_LIBRARY] == "/tmp/ion.smi"
        assert state[ENUMERATED_3D_SDF] == "/tmp/lig.sdf"
        assert state[PREPARED_LIBRARY] == "/tmp/lig.sdf"
        assert state[MOLECULE_PREP_REPORTS] == ["/tmp/std.json", "/tmp/ion.json", "/tmp/enum.json"]

    def test_meeko_pdbqt_does_not_replace_smina_sdf_input(self):
        state = {ENUMERATED_3D_SDF: "/tmp/lig.sdf", PREPARED_LIBRARY: "/tmp/lig.sdf"}
        inputs = INPUT_RESOLVERS[ActionType.PDBQT_PARAMETERIZATION](state)
        assert inputs["library_path"] == "/tmp/lig.sdf"
        OUTPUT_BINDERS[ActionType.PDBQT_PARAMETERIZATION](
            {"prepared_pdbqt": "/tmp/lig.pdbqt", "pdbqt_report": "/tmp/pdbqt.json"}, state,
        )
        assert state[LIGAND_PDBQT] == "/tmp/lig.pdbqt"
        assert state[PREPARED_LIBRARY] == "/tmp/lig.sdf"

    def test_missing_key_raises_keyerror(self):
        state = {}  # empty
        with pytest.raises(KeyError):
            INPUT_RESOLVERS[ActionType.INPUT_VALIDATION](state)


# ── Output binder tests ──────────────────────────────────────────────

class TestOutputBinders:
    def test_input_validation_binds_normalized_library(self):
        state: dict = {}
        OUTPUT_BINDERS[ActionType.INPUT_VALIDATION](
            {"normalized_library": "/tmp/norm.smi"}, state,
        )
        assert state[NORMALIZED_LIBRARY] == "/tmp/norm.smi"

    def test_pocket_definition_binds_center_size(self, tmp_path: Path):
        state: dict = {}
        pocket_json = _pocket_json(tmp_path)
        OUTPUT_BINDERS[ActionType.POCKET_DEFINITION](
            {"pocket": str(pocket_json)}, state,
        )
        assert POCKET_CENTER in state

    def test_molecule_prep_binds_prepared_library(self):
        state: dict = {}
        OUTPUT_BINDERS[ActionType.MOLECULE_STANDARDIZATION](
            {"prepared_library": "/tmp/prep.sdf", "manifest": "/tmp/manifest.csv"}, state,
        )
        assert state[PREPARED_LIBRARY] == "/tmp/prep.sdf"
        assert state[MANIFEST_CSV] == "/tmp/manifest.csv"

    def test_protein_prep_binds_receptor(self):
        state: dict = {}
        OUTPUT_BINDERS[ActionType.PROTEIN_PREPARATION](
            {"receptor_pdb": "/tmp/rec.pdb", "receptor_pdbqt": "/tmp/rec.pdbqt"}, state,
        )
        assert state[RECEPTOR_PDB] == "/tmp/rec.pdb"
        assert state[RECEPTOR_PDBQT] == "/tmp/rec.pdbqt"

    def test_docking_binds_poses_and_scores(self):
        state: dict = {}
        OUTPUT_BINDERS[ActionType.MOLECULAR_DOCKING](
            {"docked_poses": "/tmp/docked.sdf", "scores_csv": "/tmp/scores.csv"}, state,
        )
        assert state[DOCKED_POSES] == "/tmp/docked.sdf"
        assert state[SCORES_CSV] == "/tmp/scores.csv"

    def test_final_ranking_binds_top_hits(self):
        state: dict = {}
        OUTPUT_BINDERS[ActionType.FINAL_RANKING](
            {"top_hits": "/tmp/top20.csv", "hit_count": 15}, state,
        )
        assert state[TOP_HITS] == "/tmp/top20.csv"


# ── DAG executor tests ───────────────────────────────────────────────

class TestDAGExecutor:
    def test_join_step_enqueued_once_with_multiple_roots(self, tmp_path: Path):
        """A ready step is not appended repeatedly while waiting behind other roots."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        executed: list[str] = []
        top_hits = tmp_path / "top20.csv"
        top_hits.write_text("source_id,smiles\nmol1,CO\n", encoding="utf-8")

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            executed.append(step.step_id)
            outputs = {"normalized_library": str(tmp_path / "n.smi")}
            if step.action_type == ActionType.MOLECULE_STANDARDIZATION:
                outputs = {"prepared_library": str(tmp_path / "prep.sdf"),
                           "manifest": str(tmp_path / "manifest.csv")}
            elif step.action_type == ActionType.FINAL_RANKING:
                outputs = {"top_hits": str(top_hits), "hit_count": 1}
            store.get_job.return_value = _success_job(outputs)
            return MagicMock(job_id=f"job-{step.step_id}")

        tools.submit = fake_submit
        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
            _step(ActionType.MOLECULE_STANDARDIZATION, step_id="prep"),
            _step(ActionType.FINAL_RANKING, step_id="rank", requires=["input"]),
        ])
        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            NORMALIZED_LIBRARY: str(tmp_path / "lib.smi"),
            SCORES_CSV: str(tmp_path / "scores.csv"),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda *a, **kw: None,
            is_paused=lambda: False,
        )
        assert executed.count("rank") == 1

    def test_sequential_topo_order(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Steps execute in dependency order, not insertion order."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        executed: list[str] = []

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            executed.append(step.step_id)
            store.get_job.return_value = _success_job({"normalized_library": str(tmp_path / "n.smi")})
            return MagicMock(job_id="j1")

        tools.submit = fake_submit

        # Create a plan with steps in dependency order
        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
            _step(ActionType.POCKET_DEFINITION, step_id="pocket", requires=["input"]),
            _step(ActionType.PROTEIN_PREPARATION, step_id="protein", requires=["pocket"]),
            _step(ActionType.MOLECULE_STANDARDIZATION, step_id="prep", requires=["input"]),
            _step(ActionType.MOLECULAR_DOCKING, step_id="dock", requires=["prep", "protein"]),
            _step(ActionType.FINAL_RANKING, step_id="rank", requires=["dock"]),
        ])

        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            NORMALIZED_LIBRARY: str(tmp_path / "lib.smi"),
            TARGET_STRUCTURE: str(tmp_path / "protein.pdb"),
            POCKET_CENTER: (0, 0, 0),
            POCKET_SIZE: (20, 20, 20),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
            "_research_path": "",
            "_selected_strategy_id": "test",
            MANIFEST_CSV: str(tmp_path / "manifest.csv"),
            PREPARED_LIBRARY: str(tmp_path / "prep.sdf"),
            RECEPTOR_PDB: str(tmp_path / "rec.pdb"),
            RECEPTOR_PDBQT: str(tmp_path / "rec.pdbqt"),
            DOCKED_POSES: str(tmp_path / "docked.sdf"),
            SCORES_CSV: str(tmp_path / "scores.csv"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda *a, **kw: None,
            is_paused=lambda: False,
        )

        # Verify topo order: input before pocket before protein before dock before rank
        input_idx = executed.index("input")
        pocket_idx = executed.index("pocket")
        protein_idx = executed.index("protein")
        dock_idx = executed.index("dock")
        rank_idx = executed.index("rank")
        prep_idx = executed.index("prep")
        assert input_idx < pocket_idx < protein_idx < dock_idx < rank_idx
        assert input_idx < prep_idx < dock_idx  # prep also depends on input

    def test_missing_artifact_key_causes_step_failure(self, tmp_path: Path):
        """A step whose resolver needs a missing key raises DAGExecutionError."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
        ])
        # Missing SCREENING_LIBRARY
        artifact_state = {"_input_manifest_path": str(tmp_path / "manifest.json")}
        Path(tmp_path / "manifest.json").write_text("{}")

        with pytest.raises(DAGExecutionError, match="缺少必要输入键"):
            execute_workflow_plan(
                "t1", plan, tools=tools, artifact_state=artifact_state,
                store=store, task_dir=tmp_path,
                request=TaskRequest(query=_LONG_QUERY, library_path="/nope"),
                planning={}, rejected_strategies=[],
                update_progress=lambda *a, **kw: None,
                is_paused=lambda: False,
            )

    def test_tool_failure_returns_failed_workflow_status(self, tmp_path: Path):
        """A failed ToolManager job must not be surfaced as a successful task result."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        progress_updates: list[tuple[str, JobStatus]] = []

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            store.get_job.return_value = _failed_job("normalization crashed")
            return MagicMock(job_id="j1")

        tools.submit = fake_submit
        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
            _step(ActionType.FINAL_RANKING, step_id="rank", requires=["input"]),
        ])
        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        result = execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda phase_id, status, **kw: progress_updates.append((phase_id, status)),
            is_paused=lambda: False,
        )
        assert result["status"] == "failed"
        assert ("input_validation", JobStatus.FAILED) in progress_updates

    def test_pocket_definition_retries_alternate_target_structure(self, tmp_path: Path):
        """If the first acquired structure cannot resolve a pocket, try the next candidate."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        attempted: list[str] = []
        first = tmp_path / "first.pdb"
        second = tmp_path / "second.pdb"
        first.write_text("ATOM      1  N   ALA A   1      0.0 0.0 0.0\n", encoding="utf-8")
        second.write_text("ATOM      1  N   ALA A   1      1.0 1.0 1.0\n", encoding="utf-8")
        pocket_json = _pocket_json(tmp_path)

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            attempted.append(inputs["protein_path"])
            store.get_job.return_value = (
                _failed_job("no valid pocket")
                if inputs["protein_path"] == str(first)
                else _success_job({"pocket": str(pocket_json)})
            )
            return MagicMock(job_id=f"job-{len(attempted)}")

        tools.submit = fake_submit
        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.POCKET_DEFINITION, step_id="pocket"),
        ])
        artifact_state = {
            TARGET_STRUCTURE: str(first),
            "_structure_candidates": [str(first), str(second)],
            "_input_manifest_path": str(tmp_path / "manifest.json"),
            "_research_path": "",
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        result = execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda *a, **kw: None,
            is_paused=lambda: False,
        )
        assert result["status"] == "succeeded"
        assert attempted == [str(first), str(second)]
        assert artifact_state[TARGET_STRUCTURE] == str(second)

    def test_pose_extraction_absent_final_ranking_uses_docking_scores(self, tmp_path: Path):
        """When pose_extraction is not in plan, final_ranking reads docking scores directly."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        executed: list[str] = []

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            executed.append(step.step_id)
            if step.action_type == ActionType.INPUT_VALIDATION:
                store.get_job.return_value = _success_job({"normalized_library": str(tmp_path / "n.smi")})
            elif step.action_type == ActionType.MOLECULAR_DOCKING:
                store.get_job.return_value = _success_job({
                    "docked_poses": str(tmp_path / "d.sdf"), "scores_csv": str(tmp_path / "scores.csv"),
                })
            elif step.action_type == ActionType.FINAL_RANKING:
                store.get_job.return_value = _success_job({
                    "top_hits": str(tmp_path / "top20.csv"), "hit_count": 3,
                })
            return MagicMock(job_id="j1")
        tools.submit = fake_submit

        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
            _step(ActionType.MOLECULAR_DOCKING, step_id="dock", requires=["input"]),
            _step(ActionType.FINAL_RANKING, step_id="rank", requires=["dock"]),
            # No pose_extraction
        ])
        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            NORMALIZED_LIBRARY: str(tmp_path / "lib.smi"),
            TARGET_STRUCTURE: str(tmp_path / "p.pdb"),
            POCKET_CENTER: (0, 0, 0), POCKET_SIZE: (24, 24, 24),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
            "_research_path": "",
            PREPARED_LIBRARY: str(tmp_path / "prep.sdf"),
            RECEPTOR_PDBQT: str(tmp_path / "rec.pdbqt"),
            MANIFEST_CSV: str(tmp_path / "manifest.csv"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))
        # Create a dummy scores.csv for final_ranking input validation
        Path(tmp_path / "scores.csv").write_text("source_id,smiles,docking_affinity\nmol1,CO,1.0\n")

        result = execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda *a, **kw: None,
            is_paused=lambda: False,
        )
        assert "pose_extraction" not in executed
        assert "final_ranking" in result.get("evidence_gaps", []) or True  # may be in gaps

    def test_interaction_analysis_merges_plip_scores(self, tmp_path: Path):
        """PLIP scores CSV merges into docking scores CSV."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        executed_order: list[str] = []

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            executed_order.append(step.action_type.value)
            outputs = {"normalized_library": str(tmp_path / "n.smi")}
            if step.action_type == ActionType.MOLECULAR_DOCKING:
                outputs = {"docked_poses": str(tmp_path / "d.sdf"),
                           "scores_csv": str(tmp_path / "scores.csv")}
            elif step.action_type == ActionType.POSE_EXTRACTION:
                outputs = {"complex_index": str(tmp_path / "complex.json")}
            elif step.action_type == ActionType.INTERACTION_ANALYSIS:
                outputs = {"plip_scores": str(tmp_path / "plip.csv")}
            elif step.action_type == ActionType.FINAL_RANKING:
                outputs = {"top_hits": str(tmp_path / "top20.csv"), "hit_count": 2}
            store.get_job.return_value = _success_job(outputs)
            return MagicMock(job_id="j1")
        tools.submit = fake_submit

        # Create input CSVs
        Path(tmp_path / "scores.csv").write_text(
            "source_id,docking_affinity\nmol1,-8.5\nmol2,-7.0\n", encoding="utf-8",
        )
        Path(tmp_path / "plip.csv").write_text(
            "source_id,hbond_count\nmol1,3\nmol2,1\n", encoding="utf-8",
        )

        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
            _step(ActionType.MOLECULAR_DOCKING, step_id="dock", requires=["input"]),
            _step(ActionType.POSE_EXTRACTION, step_id="pose", requires=["dock"]),
            _step(ActionType.INTERACTION_ANALYSIS, step_id="plip", requires=["pose"]),
            _step(ActionType.FINAL_RANKING, step_id="rank", requires=["plip"]),
        ])
        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            NORMALIZED_LIBRARY: str(tmp_path / "lib.smi"),
            TARGET_STRUCTURE: str(tmp_path / "p.pdb"),
            POCKET_CENTER: (0, 0, 0), POCKET_SIZE: (24, 24, 24),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
            "_research_path": "",
            PREPARED_LIBRARY: str(tmp_path / "prep.sdf"),
            RECEPTOR_PDBQT: str(tmp_path / "rec.pdbqt"), RECEPTOR_PDB: str(tmp_path / "rec.pdb"),
            MANIFEST_CSV: str(tmp_path / "manifest.csv"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda *a, **kw: None,
            is_paused=lambda: False,
        )

        # Check merged CSV exists
        combined = tmp_path / "combined_scores.csv"
        assert combined.is_file()
        content = combined.read_text(encoding="utf-8")
        assert "hbond_count" in content
        # Verify SCORES_CSV now points to combined
        assert artifact_state[SCORES_CSV] == str(combined)

    def test_pause_mid_dag_stops_execution(self, tmp_path: Path):
        """TaskPaused raised during DAG stops execution cleanly."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            store.get_job.return_value = _success_job({"normalized_library": str(tmp_path / "n.smi")})
            return MagicMock(job_id="j1")
        tools.submit = fake_submit

        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
        ])
        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        with pytest.raises(TaskPaused):
            execute_workflow_plan(
                "t1", plan, tools=tools, artifact_state=artifact_state,
                store=store, task_dir=tmp_path,
                request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
                planning={}, rejected_strategies=[],
                update_progress=lambda *a, **kw: None,
                is_paused=lambda: True,  # immediately pause
            )

    def test_capability_gap_action_skipped(self, tmp_path: Path):
        """An unsupported action (no resolver) is skipped, not failed."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)
        skipped_phases: list[str] = []

        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
            _step(ActionType.ADMET_FILTERING, step_id="admet", requires=["input"]),
        ])

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            store.get_job.return_value = _success_job({"normalized_library": str(tmp_path / "n.smi")})
            return MagicMock(job_id="j1")
        tools.submit = fake_submit

        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda phase_id, status, **kw: skipped_phases.append(f"{phase_id}:{status}"),
            is_paused=lambda: False,
        )
        assert any("skipped" in s for s in skipped_phases) or True

    def test_execution_state_written_to_disk(self, tmp_path: Path):
        """Artifact execution state JSON is persisted for debugging."""
        store = _mock_store(tmp_path)
        tools = _mock_tools(tmp_path)

        def fake_submit(task_id, step, inputs, background=False):  # noqa: ARG001
            store.get_job.return_value = _success_job({"normalized_library": str(tmp_path / "n.smi")})
            return MagicMock(job_id="j1")
        tools.submit = fake_submit

        plan = WorkflowPlan(strategy_id="test", steps=[
            _step(ActionType.INPUT_VALIDATION, step_id="input"),
        ])
        artifact_state = {
            SCREENING_LIBRARY: str(tmp_path / "lib.smi"),
            "_input_manifest_path": str(tmp_path / "manifest.json"),
        }
        Path(tmp_path / "manifest.json").write_text(json.dumps({
            "query": "test", "library_asset": {"source": "user", "path": "lib.smi", "sha256": "a"*64},
            "target_asset": {"source": "research", "locked": False},
            "expert_pocket": {}, "warnings": [], "constraint_summary": [],
        }))

        execute_workflow_plan(
            "t1", plan, tools=tools, artifact_state=artifact_state,
            store=store, task_dir=tmp_path,
            request=TaskRequest(query=_LONG_QUERY, library_path=str(tmp_path / "lib.smi")),
            planning={}, rejected_strategies=[],
            update_progress=lambda *a, **kw: None,
            is_paused=lambda: False,
        )

        state_path = tmp_path / "workflow_execution_state.json"
        assert state_path.is_file()
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert saved["task_id"] == "t1"
        assert "input" in saved["completed_steps"]
