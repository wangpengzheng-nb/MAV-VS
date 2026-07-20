from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from autovs.schemas import (
    ActionType, ArtifactRef, ResourceProfile, WorkflowPlan, WorkflowStep,
)


ALIASES = {
    "library_preparation": ActionType.MOLECULE_STANDARDIZATION,
    "physicochemical_filtering": ActionType.PHYSICOCHEMICAL_FILTERING,
    "protein_preparation": ActionType.PROTEIN_PREPARATION,
    "binding_site_detection": ActionType.POCKET_DEFINITION,
    "molecular_docking": ActionType.MOLECULAR_DOCKING,
    "interaction_analysis": ActionType.INTERACTION_ANALYSIS,
    "admet_filtering": ActionType.ADMET_FILTERING,
    "diversity_selection": ActionType.DIVERSITY_SELECTION,
    "molecular_dynamics": ActionType.MOLECULAR_DYNAMICS,
    "final_ranking": ActionType.FINAL_RANKING,
    "report_generation": ActionType.REPORT_GENERATION,
}

UNSUPPORTED_V1 = {
    "covalent_docking", "free_energy_calculation", "fragment_growing",
    "generative_design", "water_analysis", "shape_matching",
    "pharmacophore_screening", "similarity_screening", "machine_learning_scoring",
    "visual_inspection", "consensus_scoring",
}


def _slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned[:64] or fallback


def compile_strategy(strategy: dict[str, Any]) -> WorkflowPlan:
    """Normalize an evolved/legacy strategy into strict WorkflowPlan v1.

    Unsupported scientific actions are rejected instead of silently mocked.
    Mandatory reproducibility steps are added around the strategy-defined core.
    """
    raw_steps = strategy.get("updated_pipeline") or strategy.get("pipeline") or strategy.get("pipeline_steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("strategy has no pipeline")
    converted: list[WorkflowStep] = []
    previous: str | None = None
    for index, raw in enumerate(raw_steps, 1):
        raw_action = str(raw.get("action_type", "")).strip()
        if raw_action in UNSUPPORTED_V1:
            raise ValueError(f"unsupported v1 action: {raw_action}")
        try:
            action = ALIASES[raw_action] if raw_action in ALIASES else ActionType(raw_action)
        except ValueError as exc:
            raise ValueError(f"unknown action_type: {raw_action}") from exc
        step_id = _slug(str(raw.get("step_id") or raw.get("id") or f"strategy-{index}"), f"strategy-{index}")
        params = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else raw.get("params", {})
        executor = "python"
        environment = None
        gpu = False
        if action == ActionType.MOLECULAR_DOCKING:
            executor, environment = "slurm", "smina_stage2"
        elif action == ActionType.INTERACTION_ANALYSIS:
            executor, environment = "conda", "plip"
        elif action == ActionType.ADMET_FILTERING:
            executor, environment = "conda", "autovs-admet"
        elif action in {ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS}:
            executor, gpu = "apptainer", True
        converted.append(WorkflowStep(
            step_id=step_id, action_type=action, requires=[previous] if previous else [],
            parameters=params or {}, quality_gates=[],
            resource_profile=ResourceProfile(executor=executor, environment=environment, gpu_required=gpu),
        ))
        previous = step_id

    # Pocket resolution is a deterministic scientific preflight owned by PipelineService.
    # LLM-authored copies (and their coordinates/parameters) are never executable.
    converted = [step for step in converted if step.action_type != ActionType.POCKET_DEFINITION]

    mandatory_prefix = [
        WorkflowStep(step_id="input-validation", action_type=ActionType.INPUT_VALIDATION),
        WorkflowStep(step_id="pocket-definition", action_type=ActionType.POCKET_DEFINITION, requires=["input-validation"]),
    ]
    existing = {s.action_type for s in converted}
    if ActionType.PROTEIN_PREPARATION not in existing:
        converted.insert(0, WorkflowStep(step_id="protein-preparation", action_type=ActionType.PROTEIN_PREPARATION))
    if ActionType.MOLECULE_STANDARDIZATION not in existing:
        converted.insert(0, WorkflowStep(step_id="molecule-preparation", action_type=ActionType.MOLECULE_STANDARDIZATION))

    # Rebuild a deterministic linear dependency chain after normalization.
    ordered = mandatory_prefix + converted
    suffix = []
    if ActionType.FINAL_RANKING not in {s.action_type for s in ordered}:
        suffix.append(WorkflowStep(step_id="final-ranking", action_type=ActionType.FINAL_RANKING))
    if ActionType.REPORT_GENERATION not in {s.action_type for s in ordered}:
        suffix.append(WorkflowStep(step_id="report-generation", action_type=ActionType.REPORT_GENERATION))
    ordered.extend(suffix)
    for index, step in enumerate(ordered):
        step.requires = [ordered[index - 1].step_id] if index else []
    strategy_id = str(strategy.get("strategy_id") or strategy.get("strategy_name") or "strategy")
    return WorkflowPlan(strategy_id=strategy_id[:120], steps=ordered)


def choose_executable_strategy(ranked_names: list[str], strategies: list[dict]) -> tuple[dict, WorkflowPlan, list[dict]]:
    by_name = {str(s.get("strategy_name", "")): s for s in strategies}
    rejected: list[dict] = []
    candidates = ranked_names + [name for name in by_name if name not in ranked_names]
    for name in candidates:
        strategy = by_name.get(name)
        if not strategy:
            continue
        try:
            return strategy, compile_strategy(strategy), rejected
        except (ValueError, ValidationError) as exc:
            rejected.append({"strategy_name": name, "reason": str(exc)})
    raise ValueError(f"no executable strategy; rejected={rejected}")
