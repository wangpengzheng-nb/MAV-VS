"""DAG workflow executor with standard artifact binding.

Replaces the fixed ``_execute_core`` pipeline with a plan-driven executor that
reads ``WorkflowPlan.steps`` and their ``requires`` to determine execution order
(v1: sequential topological sort, no parallelism).

Each action resolves its inputs from an in-memory ``artifact_state`` registry
of standard keys and writes outputs back to the same registry.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from autovs.capabilities import health_report
from autovs.reporting import generate_report
from autovs.schemas import (
    ActionType, InputManifest, JobStatus, PocketResolution, TaskRequest,
    WorkflowPlan, WorkflowStep,
)
from autovs.tool_manager import ToolManager

# ── Standard artifact keys ───────────────────────────────────────────

SCREENING_LIBRARY = "screening_library"       # Path: raw input SMILES/CSV
NORMALIZED_LIBRARY = "normalized_library"      # Path: validated/repaired SMILES
TARGET_STRUCTURE = "target_structure"          # Path: PDB file
POCKET_RESOLUTION = "pocket_resolution"        # PocketResolution dict
POCKET_CENTER = "pocket_center"               # (float, float, float)
POCKET_SIZE = "pocket_size"                   # (float, float, float)
PREPARED_LIBRARY = "prepared_library"          # Path: prepared SDF
MANIFEST_CSV = "manifest_csv"                 # Path: molecule manifest CSV
RECEPTOR_PDB = "receptor_pdb"                 # Path: cleaned receptor PDB
RECEPTOR_PDBQT = "receptor_pdbqt"             # Path: receptor PDBQT
DOCKED_POSES = "docked_poses"                 # Path: docking output SDF
SCORES_CSV = "scores_csv"                     # Path: docking scores CSV
SELECTED_POSES = "selected_poses"             # Path: extracted best poses SDF
COMPLEX_INDEX = "complex_index"               # Path: complex index JSON
PLIP_SCORES = "plip_scores"                   # Path: PLIP interaction scores CSV
TOP_HITS = "top_hits"                         # Path: final ranked top-N CSV
HIT_COUNT = "hit_count"                       # int


# ── Input resolver / output binder registry ──────────────────────────

InputResolver = Callable[..., dict[str, Any]]
OutputBinder = Callable[[dict[str, Any], dict[str, Any]], None]


def _resolve_input_validation(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "protein_path": state.get(TARGET_STRUCTURE),
        "library_path": state[SCREENING_LIBRARY],
        "input_manifest_path": state["_input_manifest_path"],
    }


def _bind_input_validation(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[NORMALIZED_LIBRARY] = outputs["normalized_library"]
    state["_accepted_records"] = outputs.get("accepted_records", 0)
    state["_quarantined_records"] = outputs.get("quarantined_records", 0)


def _resolve_target_structure(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "research_path": state["_research_path"],
        "limit": kwargs.get("limit", 5),
        "selected_strategy_id": state.get("_selected_strategy_id", ""),
    }


def _bind_target_structure(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    candidates = outputs.get("candidate_structures", [])
    if candidates:
        state[TARGET_STRUCTURE] = candidates[0]
        # Store the first valid structure as target_structure
        metadata = {str(item.get("path", "")): str(item.get("pdb_id") or "")
                    for item in outputs.get("candidates", [])}
        state["_structure_candidates"] = candidates
        state["_structure_metadata"] = metadata


def _resolve_pocket_definition(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    center = kwargs.get("center")
    if center is None:
        center = state.get(POCKET_CENTER)
    size = kwargs.get("size")
    if size is None:
        size = state.get(POCKET_SIZE, (24, 24, 24))
    key_residues = kwargs.get("key_residues")
    if key_residues is None:
        key_residues = state.get("_pocket_key_residues", [])
    cocrystal_ligand = kwargs.get("cocrystal_ligand")
    if cocrystal_ligand is None:
        cocrystal_ligand = state.get("_pocket_cocrystal_ligand")
    inputs: dict[str, Any] = {
        "protein_path": state[TARGET_STRUCTURE],
        "center": center,
        "size": size,
        "key_residues": key_residues,
        "cocrystal_ligand": cocrystal_ligand,
    }
    research_path = state.get("_research_path")
    if research_path and Path(research_path).is_file():
        inputs["research_path"] = research_path
    return inputs


def _bind_pocket_definition(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    pocket_path = outputs.get("pocket")
    if pocket_path:
        resolution = PocketResolution.model_validate_json(Path(pocket_path).read_text(encoding="utf-8"))
        state[POCKET_RESOLUTION] = resolution.model_dump(mode="json")
        pocket_data = resolution.selected_pocket
        state[POCKET_CENTER] = pocket_data.center
        state[POCKET_SIZE] = pocket_data.size


def _resolve_molecule_prep(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {"library_path": state[NORMALIZED_LIBRARY]}


def _bind_molecule_prep(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[PREPARED_LIBRARY] = outputs.get("prepared_library")
    state[MANIFEST_CSV] = outputs.get("manifest")


def _resolve_protein_prep(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {"protein_path": state[TARGET_STRUCTURE]}


def _bind_protein_prep(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[RECEPTOR_PDB] = outputs.get("receptor_pdb")
    state[RECEPTOR_PDBQT] = outputs.get("receptor_pdbqt")


def _resolve_docking(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "receptor_pdbqt": state[RECEPTOR_PDBQT],
        "ligands_sdf": state[PREPARED_LIBRARY],
        "manifest_csv": state.get(MANIFEST_CSV),
        "center": state[POCKET_CENTER],
        "size": state[POCKET_SIZE],
    }


def _bind_docking(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[DOCKED_POSES] = outputs.get("docked_poses")
    state[SCORES_CSV] = outputs.get("scores_csv")


def _resolve_pose_extraction(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "receptor_pdb": state[RECEPTOR_PDB],
        "docked_poses": state[DOCKED_POSES],
        "engine": kwargs.get("engine", "smina"),
        "pose_metric": kwargs.get("pose_metric", "best_affinity"),
    }


def _bind_pose_extraction(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[COMPLEX_INDEX] = outputs.get("complex_index")


def _resolve_interaction_analysis(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "complex_index": state[COMPLEX_INDEX],
        "key_residues": kwargs.get("key_residues", []),
    }


def _bind_interaction_analysis(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[PLIP_SCORES] = outputs.get("plip_scores")


def _resolve_final_ranking(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {"scores_csv": state[SCORES_CSV]}


def _bind_final_ranking(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    state[TOP_HITS] = outputs.get("top_hits")
    state[HIT_COUNT] = outputs.get("hit_count", 0)


def _resolve_structure_analysis(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "protein_path": state[TARGET_STRUCTURE],
        "center": state.get(POCKET_CENTER, kwargs.get("center")),
        "radius": kwargs.get("radius", 8.0),
    }


def _bind_structure_analysis(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    """从结构分析输出中提取口袋残基信息。"""
    if outputs.get("pocket_residues"):
        state["_pocket_residues"] = outputs["pocket_residues"]
    if outputs.get("resolution"):
        state["_structure_resolution"] = outputs["resolution"]


def _resolve_protein_repair(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "protein_path": state[TARGET_STRUCTURE],
        "add_hydrogens": kwargs.get("add_hydrogens", True),
        "add_missing_atoms": kwargs.get("add_missing_atoms", True),
        "replace_nonstandard": kwargs.get("replace_nonstandard", True),
        "remove_heterogens": kwargs.get("remove_heterogens", True),
        "keep_chains": kwargs.get("keep_chains"),
        "remove_chains": kwargs.get("remove_chains"),
        "ph": kwargs.get("ph", 7.4),
        "long_gap_threshold": kwargs.get("long_gap_threshold", 5),
    }


def _bind_protein_repair(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    """修复后的结构更新 target_structure。"""
    if outputs.get("repaired_structure"):
        state[TARGET_STRUCTURE] = outputs["repaired_structure"]
    if outputs.get("warnings"):
        state["_repair_warnings"] = outputs["warnings"]


def _resolve_protonation(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "protein_path": state[TARGET_STRUCTURE],
        "ph": kwargs.get("ph", 7.4),
        "forcefield": kwargs.get("forcefield", "PARSE"),
        "drop_water": kwargs.get("drop_water", True),
        "nodebump": kwargs.get("nodebump", False),
        "noopt": kwargs.get("noopt", False),
        "chains": kwargs.get("chains"),
    }


def _bind_protonation(outputs: dict[str, Any], state: dict[str, Any]) -> None:
    """质子化后的结构更新 target_structure，PQR 路径供下游使用。"""
    if outputs.get("protonated_pdb"):
        state[TARGET_STRUCTURE] = outputs["protonated_pdb"]
    if outputs.get("output_pqr"):
        state["_protonated_pqr"] = outputs["output_pqr"]


# ── Registry maps ────────────────────────────────────────────────────

INPUT_RESOLVERS: dict[ActionType, InputResolver] = {
    ActionType.INPUT_VALIDATION: _resolve_input_validation,
    ActionType.TARGET_STRUCTURE_ACQUISITION: _resolve_target_structure,
    ActionType.POCKET_DEFINITION: _resolve_pocket_definition,
    ActionType.MOLECULE_STANDARDIZATION: _resolve_molecule_prep,
    ActionType.CONFORMER_GENERATION: _resolve_molecule_prep,
    ActionType.PHYSICOCHEMICAL_FILTERING: _resolve_molecule_prep,
    ActionType.PROTEIN_PREPARATION: _resolve_protein_prep,
    ActionType.MOLECULAR_DOCKING: _resolve_docking,
    ActionType.POSE_EXTRACTION: _resolve_pose_extraction,
    ActionType.INTERACTION_ANALYSIS: _resolve_interaction_analysis,
    ActionType.FINAL_RANKING: _resolve_final_ranking,
    ActionType.STRUCTURE_ANALYSIS: _resolve_structure_analysis,
    ActionType.PROTEIN_REPAIR: _resolve_protein_repair,
    ActionType.PROTONATION: _resolve_protonation,
}

OUTPUT_BINDERS: dict[ActionType, OutputBinder] = {
    ActionType.INPUT_VALIDATION: _bind_input_validation,
    ActionType.TARGET_STRUCTURE_ACQUISITION: _bind_target_structure,
    ActionType.POCKET_DEFINITION: _bind_pocket_definition,
    ActionType.MOLECULE_STANDARDIZATION: _bind_molecule_prep,
    ActionType.CONFORMER_GENERATION: _bind_molecule_prep,
    ActionType.PHYSICOCHEMICAL_FILTERING: _bind_molecule_prep,
    ActionType.PROTEIN_PREPARATION: _bind_protein_prep,
    ActionType.MOLECULAR_DOCKING: _bind_docking,
    ActionType.POSE_EXTRACTION: _bind_pose_extraction,
    ActionType.INTERACTION_ANALYSIS: _bind_interaction_analysis,
    ActionType.FINAL_RANKING: _bind_final_ranking,
    ActionType.STRUCTURE_ANALYSIS: _bind_structure_analysis,
    ActionType.PROTEIN_REPAIR: _bind_protein_repair,
    ActionType.PROTONATION: _bind_protonation,
}


# ── DAG executor ─────────────────────────────────────────────────────

# Phases that map to action types for progress reporting.
ACTION_PHASE_MAP: dict[ActionType, str] = {
    ActionType.INPUT_VALIDATION: "input_validation",
    ActionType.TARGET_STRUCTURE_ACQUISITION: "target_structure_acquisition",
    ActionType.TARGET_STRUCTURE_PREDICTION: "target_structure_acquisition",
    ActionType.POCKET_DEFINITION: "pocket_definition",
    ActionType.MOLECULE_STANDARDIZATION: "molecule_standardization",
    ActionType.CONFORMER_GENERATION: "molecule_standardization",
    ActionType.PHYSICOCHEMICAL_FILTERING: "molecule_standardization",
    ActionType.PROTEIN_PREPARATION: "protein_preparation",
    ActionType.MOLECULAR_DOCKING: "molecular_docking",
    ActionType.POSE_EXTRACTION: "pose_extraction",
    ActionType.INTERACTION_ANALYSIS: "interaction_analysis",
    ActionType.FINAL_RANKING: "final_ranking",
    ActionType.STRUCTURE_ANALYSIS: "target_structure_acquisition",
    ActionType.PROTEIN_REPAIR: "protein_preparation",
    ActionType.PROTONATION: "protein_preparation",
}


class DAGExecutionError(RuntimeError):
    """Raised when a workflow step fails within the DAG executor."""

    def __init__(self, step_id: str, action_type: ActionType, reason: str):
        super().__init__(f"[{step_id}] {action_type.value}: {reason}")
        self.step_id = step_id
        self.action_type = action_type


class TaskPaused(Exception):
    """信号：任务已暂停，DAG 执行器应优雅退出。"""
    pass


def _merge_score_csvs(primary: Path, additional: Path, output: Path) -> Path:
    """Merge additional scoring columns into the primary scores CSV."""
    with primary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with additional.open(encoding="utf-8-sig", newline="") as handle:
        extra = {row["source_id"]: row for row in csv.DictReader(handle)}
    for row in rows:
        row.update(extra.get(row.get("source_id", ""), {}))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def execute_workflow_plan(
    task_id: str,
    plan: WorkflowPlan,
    *,
    tools: ToolManager,
    artifact_state: dict[str, Any],
    store: Any,  # StateStore
    task_dir: Path,
    request: TaskRequest,
    planning: dict[str, Any],
    rejected_strategies: list[dict],
    update_progress: Callable[..., None],
    is_paused: Callable[[], bool],
) -> dict[str, Any]:
    """Execute a ``WorkflowPlan`` as a sequential DAG.

    Parameters
    ----------
    task_id : str
    plan : WorkflowPlan
        The compiled plan whose ``steps`` and their ``requires`` determine
        execution order (v1: sequential; reserved for future parallelism).
    tools : ToolManager
    artifact_state : dict
        In-memory registry of standard artifact keys.  The executor reads
        inputs from this dict and writes outputs back to it after each step.
    store : StateStore
    task_dir : Path
    request : TaskRequest
    planning : dict
        Pre-DAG planning metadata (target research, strategy rankings, etc.).
    rejected_strategies : list[dict]
    update_progress : callable
        ``(phase_id, status, **kwargs) -> None``
    is_paused : callable
        ``() -> bool`` — raises ``_TaskPaused`` internally if True.
    """

    # ── Build ready-set from topology ────────────────────────────
    completed: set[str] = set()
    ready: list[WorkflowStep] = [s for s in plan.steps if not s.requires]
    queued: set[str] = {s.step_id for s in ready}

    # Pre-populate pocket center/size from user request
    if request.pocket.center:
        artifact_state[POCKET_CENTER] = request.pocket.center
    if request.pocket.size:
        artifact_state[POCKET_SIZE] = request.pocket.size
    if request.pocket.key_residues:
        artifact_state["_pocket_key_residues"] = request.pocket.key_residues
    if request.pocket.cocrystal_ligand:
        artifact_state["_pocket_cocrystal_ligand"] = request.pocket.cocrystal_ligand

    # Internal state used by resolvers
    artifact_state["_input_manifest_path"] = request.input_manifest_path
    artifact_state["_task_dir"] = str(task_dir)

    pending = list(plan.steps)
    failed = False
    blocked_steps: set[str] = set()

    def _enqueue_newly_ready() -> None:
        for candidate in pending:
            if candidate.step_id in completed or candidate.step_id in queued:
                continue
            if set(candidate.requires).issubset(completed):
                ready.append(candidate)
                queued.add(candidate.step_id)

    while ready:
        # Take the first ready step (sequential order; later: parallel-ready subset)
        step = ready.pop(0)
        queued.discard(step.step_id)
        pending = [s for s in pending if s.step_id != step.step_id]

        # ── Skip report_generation in the tool loop; handled post-DAG ──
        if step.action_type == ActionType.REPORT_GENERATION:
            completed.add(step.step_id)
            _enqueue_newly_ready()
            continue

        # ── Pause / cancel check ──
        if is_paused():
            raise TaskPaused()

        # ── Resolve inputs ──
        resolver = INPUT_RESOLVERS.get(step.action_type)
        if resolver is None:
            # Capability gap: action has no production adapter yet
            phase_id = ACTION_PHASE_MAP.get(step.action_type)
            if phase_id:
                update_progress(
                    phase_id, JobStatus.SKIPPED,
                    message=f"{step.action_type.value} 尚无可用适配器",
                    metadata={"step_id": step.step_id, "action_type": step.action_type.value},
                )
            blocked_steps.add(step.step_id)
            failed = True
            continue

        try:
            step_inputs = resolver(artifact_state, **step.parameters)
        except KeyError as exc:
            raise DAGExecutionError(
                step.step_id, step.action_type,
                f"缺少必要输入键: {exc}；artifact_state 中可用的键: {sorted(artifact_state)}",
            ) from exc

        # ── Phase progress ──
        phase_id = ACTION_PHASE_MAP.get(step.action_type)
        if phase_id:
            update_progress(
                phase_id, JobStatus.RUNNING,
                message=f"正在执行 {step.step_id}",
                metadata={"step_id": step.step_id, "action_type": step.action_type.value},
            )

        # ── Execute via ToolManager ──
        try:
            job = tools.submit(task_id, step, step_inputs, background=False)
            completed_job = store.get_job(job.job_id)
            if not completed_job or completed_job.status != JobStatus.SUCCEEDED:
                error_msg = completed_job.message if completed_job else f"step {step.step_id} disappeared"
                if step.action_type == ActionType.POCKET_DEFINITION:
                    candidates = [str(item) for item in artifact_state.get("_structure_candidates", [])]
                    current = str(artifact_state.get(TARGET_STRUCTURE, ""))
                    for candidate in candidates:
                        if candidate == current:
                            continue
                        artifact_state[TARGET_STRUCTURE] = candidate
                        try:
                            retry_inputs = resolver(artifact_state, **step.parameters)
                        except KeyError:
                            continue
                        if phase_id:
                            update_progress(
                                phase_id, JobStatus.RUNNING,
                                message=f"正在用备选靶结构重试 {step.step_id}",
                                metadata={
                                    "step_id": step.step_id,
                                    "action_type": step.action_type.value,
                                    "target_structure": candidate,
                                },
                            )
                        retry_job = tools.submit(task_id, step, retry_inputs, background=False)
                        retry_completed = store.get_job(retry_job.job_id)
                        if retry_completed and retry_completed.status == JobStatus.SUCCEEDED:
                            job = retry_job
                            completed_job = retry_completed
                            step_inputs = retry_inputs
                            error_msg = ""
                            break
                        error_msg = (
                            retry_completed.message if retry_completed
                            else f"step {step.step_id} disappeared during retry"
                        )
                if not completed_job or completed_job.status != JobStatus.SUCCEEDED:
                    if phase_id:
                        update_progress(
                            phase_id, JobStatus.FAILED,
                            message=f"工具步骤 {step.step_id} 失败",
                            error=error_msg,
                            metadata={"step_id": step.step_id, "job_id": job.job_id,
                                      "action_type": step.action_type.value},
                        )
                    failed = True
                    blocked_steps.add(step.step_id)
                    break
            outputs = json.loads(completed_job.message)
        except Exception as exc:
            if phase_id:
                update_progress(
                    phase_id, JobStatus.FAILED,
                    message=f"工具步骤 {step.step_id} 失败",
                    error=str(exc),
                    metadata={"step_id": step.step_id, "action_type": step.action_type.value},
                )
            raise DAGExecutionError(step.step_id, step.action_type, str(exc)) from exc

        # ── Bind outputs ──
        binder = OUTPUT_BINDERS.get(step.action_type)
        if binder:
            binder(outputs, artifact_state)

        if phase_id:
            update_progress(
                phase_id, JobStatus.SUCCEEDED,
                message=f"已完成 {step.step_id}",
                metadata={"step_id": step.step_id, "job_id": job.job_id,
                          "action_type": step.action_type.value},
            )

        completed.add(step.step_id)

        # ── Post-DAG: merge PLIP scores into docking scores ──
        if step.action_type == ActionType.INTERACTION_ANALYSIS and PLIP_SCORES in artifact_state:
            scores_csv = artifact_state.get(SCORES_CSV)
            plip_csv = artifact_state[PLIP_SCORES]
            if scores_csv and Path(scores_csv).is_file() and Path(plip_csv).is_file():
                merged = _merge_score_csvs(Path(scores_csv), Path(plip_csv),
                                           task_dir / "combined_scores.csv")
                artifact_state[SCORES_CSV] = str(merged)

        # ── Enqueue steps whose dependencies are now satisfied ──
        _enqueue_newly_ready()

    # ── Mark remaining pending steps as skipped ──
    for step in pending:
        phase_id = ACTION_PHASE_MAP.get(step.action_type)
        if phase_id:
            update_progress(
                phase_id, JobStatus.SKIPPED,
                message=(
                    "上游能力缺口，未执行" if set(step.requires) & blocked_steps
                    else "上游阶段失败，未执行" if failed
                    else "未包含在本次可执行策略中"
                ),
                metadata={"step_id": step.step_id, "action_type": step.action_type.value},
            )

    # ── Evidence gaps ──
    executed = {s.action_type for s in plan.steps if s.step_id in completed}
    evidence_gaps: list[str] = []
    for action in (ActionType.INTERACTION_ANALYSIS, ActionType.ADMET_FILTERING,
                   ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS):
        if action not in executed:
            evidence_gaps.append(action.value)

    # ── Report generation (internal, not via ToolManager) ──
    manifest = InputManifest.model_validate_json(
        Path(request.input_manifest_path or "").read_text(encoding="utf-8")
    ) if request.input_manifest_path else None

    top_hits_path = artifact_state.get(TOP_HITS)
    top_hits: list[dict] = []
    if top_hits_path and Path(top_hits_path).is_file():
        with Path(top_hits_path).open(encoding="utf-8-sig", newline="") as handle:
            top_hits = list(csv.DictReader(handle))

    pocket_resolution = artifact_state.get(POCKET_RESOLUTION, {})

    update_progress("report_generation", JobStatus.RUNNING,
                    message="正在汇总结果、版本与可追溯证据")

    artifacts = store.list_artifacts(task_id)
    reports = generate_report(
        task_id, task_dir,
        request=request.model_dump(mode="json"),
        plan=plan.model_dump(mode="json"),
        results=top_hits,
        rejected_strategies=rejected_strategies,
        health=health_report(tools.settings),
        jobs=store.list_jobs(task_id),
        artifacts=artifacts,
        pocket_resolution=pocket_resolution,
        input_manifest=manifest.model_dump(mode="json") if manifest else {},
        candidate_strategies=planning.get("evolved_strategies", []),
    )
    for name, raw_path in reports.items():
        path = Path(raw_path)
        store.add_artifact(task_id, None, name, path, path.suffix.lstrip(".").upper(),
                           hashlib.sha256(path.read_bytes()).hexdigest())

    update_progress("report_generation", JobStatus.SUCCEEDED,
                    message="可复现报告已生成")

    # ── Persist execution state for debugging ──
    exec_state = {
        "task_id": task_id,
        "completed_steps": sorted(completed),
        "artifact_state": {
            k: str(v) for k, v in artifact_state.items()
            if not k.startswith("_")
        },
    }
    (task_dir / "workflow_execution_state.json").write_text(
        json.dumps(exec_state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    state_path = task_dir / "workflow_execution_state.json"
    if state_path.is_file():
        store.add_artifact(task_id, None, "workflow_execution_state", state_path, "JSON",
                           hashlib.sha256(state_path.read_bytes()).hexdigest())

    return {
        "task_id": task_id,
        "status": "succeeded" if not failed else "failed",
        "top_hits": top_hits,
        "reports": reports,
        "workflow_plan": str(task_dir / "workflow_plan.json"),
        "evidence_gaps": evidence_gaps,
        "rejected_strategies": rejected_strategies,
        "planning": planning,
        "candidate_strategies": planning.get("evolved_strategies", []),
        "pocket_resolution": pocket_resolution,
        "input_manifest": manifest.model_dump(mode="json") if manifest else {},
    }
