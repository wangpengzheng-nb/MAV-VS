from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from autovs.schemas import (
    ActionType, ArtifactRef, InputManifest, ResourceProfile, WorkflowPlan, WorkflowStep,
)


ALIASES = {
    "target_structure_prediction": ActionType.TARGET_STRUCTURE_PREDICTION,
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
    "structure_analysis": ActionType.STRUCTURE_ANALYSIS,
    "protein_repair": ActionType.PROTEIN_REPAIR,
    "protonation": ActionType.PROTONATION,
    "molecule_standardization_v2": ActionType.MOLECULE_STANDARDIZATION_V2,
    "ligand_3d_enumeration": ActionType.LIGAND_3D_ENUMERATION,
    "ionization_enumeration": ActionType.IONIZATION_ENUMERATION,
    "pdbqt_parameterization": ActionType.PDBQT_PARAMETERIZATION,
    "format_conversion": ActionType.FORMAT_CONVERSION,
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


def _validate_asset_bindings(strategy: dict[str, Any], manifest: InputManifest | None) -> None:
    if manifest is None:
        return
    steps = strategy.get("updated_pipeline") or strategy.get("pipeline") or strategy.get("pipeline_steps") or []
    external_libraries = re.compile(r"\b(zinc|enamine|chembridge|chemdiv|pubchem|mcule)\b", re.IGNORECASE)
    absolute_path = re.compile(r"^(?:/|~[/\\]|[A-Za-z]:[/\\])")
    for raw in steps if isinstance(steps, list) else []:
        action = str(raw.get("action_type", ""))
        if action == ActionType.TARGET_STRUCTURE_ACQUISITION.value:
            raise ValueError("target_structure_acquisition is service-owned and cannot be authored by a strategy")
        binding_text = str({
            "action_type": action, "description": raw.get("description", ""),
            "input": raw.get("input", raw.get("inputs", "")),
            "parameters": raw.get("parameters", raw.get("params", {})),
        })
        if external_libraries.search(binding_text):
            raise ValueError("strategy attempts to replace locked screening_library with an external library")
        for value in _walk_strings({"parameters": raw.get("parameters", raw.get("params", {})),
                                    "inputs": raw.get("input", raw.get("inputs", {}))}):
            if absolute_path.search(value) or value.lower().startswith(("http://", "https://")):
                raise ValueError("strategy inputs may not contain absolute paths or external URLs")
        if manifest.target_asset.locked and action in {"structure_download", "pdb_download", "structure_acquisition"}:
            raise ValueError("strategy attempts to replace locked uploaded target_structure")
        if manifest.target_asset.locked and re.search(r"\b(download|fetch|rcsb)\b", binding_text, re.IGNORECASE) \
                and re.search(r"\b(pdb|structure|target)\b", binding_text, re.IGNORECASE):
            raise ValueError("strategy attempts to replace locked uploaded target_structure")


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _symbolic_inputs(action: ActionType) -> list[ArtifactRef]:
    if action in {ActionType.MOLECULE_STANDARDIZATION, ActionType.CONFORMER_GENERATION,
                  ActionType.PHYSICOCHEMICAL_FILTERING, ActionType.DIVERSITY_SELECTION}:
        return [ArtifactRef(name="screening_library", format="strict_smi_v1")]
    if action == ActionType.MOLECULE_STANDARDIZATION_V2:
        return [ArtifactRef(name="screening_library", format="strict_smi_v1")]
    if action == ActionType.IONIZATION_ENUMERATION:
        return [ArtifactRef(name="standardized_library", format="strict_smi_v1")]
    if action == ActionType.LIGAND_3D_ENUMERATION:
        return [ArtifactRef(name="standardized_or_ionized_library", format="strict_smi_v1")]
    if action == ActionType.PDBQT_PARAMETERIZATION:
        return [ArtifactRef(name="enumerated_3d_sdf", format="SDF")]
    if action == ActionType.FORMAT_CONVERSION:
        return [ArtifactRef(name="molecule_artifact", format="SMI/SDF/PDBQT")]
    if action in {ActionType.PROTEIN_PREPARATION, ActionType.POCKET_DEFINITION}:
        return [ArtifactRef(name="target_structure", format="PDB")]
    if action in {ActionType.STRUCTURE_ANALYSIS, ActionType.PROTEIN_REPAIR, ActionType.PROTONATION}:
        return [ArtifactRef(name="target_structure", format="PDB")]
    if action == ActionType.MOLECULAR_DOCKING:
        return [ArtifactRef(name="target_structure", format="PDB"),
                ArtifactRef(name="prepared_library", format="SDF")]
    return []


def _produces_ligand_sdf(step: WorkflowStep) -> bool:
    if step.action_type in {
        ActionType.MOLECULE_STANDARDIZATION,
        ActionType.CONFORMER_GENERATION,
        ActionType.PHYSICOCHEMICAL_FILTERING,
        ActionType.LIGAND_3D_ENUMERATION,
    }:
        return True
    return (
        step.action_type == ActionType.FORMAT_CONVERSION
        and str(step.parameters.get("output_format", "")).lower() == "sdf"
    )


def _insert_before_first(actions: list[WorkflowStep], target: ActionType, step: WorkflowStep) -> None:
    for index, existing in enumerate(actions):
        if existing.action_type == target:
            actions.insert(index, step)
            return
    actions.append(step)


def validate_workflow_bindings(plan: WorkflowPlan, manifest: InputManifest) -> None:
    for step in plan.steps:
        if manifest.target_asset.locked and step.action_type == ActionType.TARGET_STRUCTURE_ACQUISITION:
            raise ValueError("uploaded target_structure is locked; acquisition is forbidden")
        for artifact in step.inputs:
            if artifact.path is not None:
                raise ValueError(f"workflow input {artifact.name} may not contain an LLM-authored path")
        for value in _walk_strings(step.parameters):
            if re.match(r"^(?:/|~[/\\]|[A-Za-z]:[/\\])", value) or value.lower().startswith(("http://", "https://")):
                raise ValueError("workflow parameters may not contain absolute paths or external URLs")


def compile_strategy(strategy: dict[str, Any], *, input_manifest: InputManifest | None = None) -> WorkflowPlan:
    """Normalize an evolved/legacy strategy into strict WorkflowPlan v1.

    Unsupported scientific actions are rejected instead of silently mocked.
    Mandatory reproducibility steps are added around the strategy-defined core.
    """
    _validate_asset_bindings(strategy, input_manifest)
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
        if action == ActionType.TARGET_STRUCTURE_PREDICTION:
            raise ValueError(
                "capability gap: target_structure_prediction requires an AlphaFold/Boltz adapter, "
                "which is not configured yet"
            )
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
            inputs=_symbolic_inputs(action), parameters=params or {}, quality_gates=[],
            resource_profile=ResourceProfile(executor=executor, environment=environment, gpu_required=gpu),
        ))
        previous = step_id

    # Pocket resolution, structure acquisition, and input validation are service-owned.
    # LLM-authored copies (and their coordinates/parameters) are never executable.
    converted = [step for step in converted
                 if step.action_type not in {ActionType.POCKET_DEFINITION,
                                              ActionType.TARGET_STRUCTURE_ACQUISITION,
                                              ActionType.INPUT_VALIDATION}]

    requires_docking = any(step.action_type == ActionType.MOLECULAR_DOCKING for step in converted)

    mandatory_prefix = [WorkflowStep(
        step_id="input-validation", action_type=ActionType.INPUT_VALIDATION,
        inputs=[ArtifactRef(name="screening_library", format="strict_smi_v1")],
    )]
    if input_manifest is not None and input_manifest.target_asset.source == "research":
        mandatory_prefix.append(WorkflowStep(
            step_id="target-structure-acquisition", action_type=ActionType.TARGET_STRUCTURE_ACQUISITION,
            requires=[mandatory_prefix[-1].step_id],
        ))
    if requires_docking:
        mandatory_prefix.append(WorkflowStep(
            step_id="pocket-definition", action_type=ActionType.POCKET_DEFINITION,
            requires=[mandatory_prefix[-1].step_id], inputs=_symbolic_inputs(ActionType.POCKET_DEFINITION),
        ))
        before_docking = []
        for step in converted:
            before_docking.append(step)
            if step.action_type == ActionType.MOLECULAR_DOCKING:
                break
        if not any(step.action_type == ActionType.PROTEIN_PREPARATION for step in before_docking):
            _insert_before_first(
                converted, ActionType.MOLECULAR_DOCKING,
                WorkflowStep(step_id="protein-preparation", action_type=ActionType.PROTEIN_PREPARATION,
                             inputs=_symbolic_inputs(ActionType.PROTEIN_PREPARATION)),
            )
        before_docking = []
        for step in converted:
            if step.action_type == ActionType.MOLECULAR_DOCKING:
                break
            before_docking.append(step)
        if not any(_produces_ligand_sdf(step) for step in before_docking):
            _insert_before_first(
                converted, ActionType.MOLECULAR_DOCKING,
                WorkflowStep(step_id="molecule-preparation", action_type=ActionType.MOLECULE_STANDARDIZATION,
                             inputs=_symbolic_inputs(ActionType.MOLECULE_STANDARDIZATION)),
            )

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
    plan = WorkflowPlan(strategy_id=strategy_id[:120], steps=ordered)
    if input_manifest is not None:
        validate_workflow_bindings(plan, input_manifest)
    return plan


def _adapt_strategy_to_manifest(strategy: dict[str, Any], input_manifest: InputManifest | None) -> dict[str, Any]:
    """Drop structure-prediction gaps when the user already supplied a locked target structure."""
    if input_manifest is None or not input_manifest.target_asset.locked or not input_manifest.target_asset.path:
        return strategy
    adapted = dict(strategy)
    for key in ("pipeline", "updated_pipeline", "pipeline_steps"):
        steps = adapted.get(key)
        if isinstance(steps, list):
            adapted[key] = [
                step for step in steps
                if str(step.get("action_type", "")) != ActionType.TARGET_STRUCTURE_PREDICTION.value
            ]
    missing = [
        item for item in adapted.get("missing_capabilities", [])
        if not str(item).startswith(f"{ActionType.TARGET_STRUCTURE_PREDICTION.value}:")
    ]
    required = [
        item for item in adapted.get("required_capabilities", [])
        if str(item) != ActionType.TARGET_STRUCTURE_PREDICTION.value
    ]
    adapted["missing_capabilities"] = missing
    adapted["required_capabilities"] = required
    profile = adapted.get("target_profile")
    if isinstance(profile, dict):
        adapted["target_profile"] = {**profile, "has_experimental_structure": True}
    if not missing and adapted.get("execution_status") in {"partially_executable", "future_capability_required"}:
        adapted["execution_status"] = "currently_executable"
    return adapted


def choose_executable_strategy(ranked_names: list[str], strategies: list[dict], *,
                               input_manifest: InputManifest | None = None) -> tuple[dict, WorkflowPlan, list[dict]]:
    by_name = {str(s.get("strategy_name", "")): _adapt_strategy_to_manifest(s, input_manifest) for s in strategies}
    rejected: list[dict] = []
    candidates = ranked_names + [name for name in by_name if name not in ranked_names]
    for name in candidates:
        strategy = by_name.get(name) or {}
        if strategy.get("execution_status") in {"partially_executable", "future_capability_required"}:
            rejected.append({
                "strategy_name": name,
                "reason": "strategy requires capabilities that are not executable in the current pipeline",
                "execution_status": strategy.get("execution_status"),
                "missing_capabilities": strategy.get("missing_capabilities", []),
            })
    candidates.sort(key=lambda name: 0 if by_name.get(name, {}).get("execution_status", "currently_executable") == "currently_executable" else 1)
    for name in candidates:
        strategy = by_name.get(name)
        if not strategy:
            continue
        if strategy.get("execution_status") in {"partially_executable", "future_capability_required"}:
            continue
        try:
            return strategy, compile_strategy(strategy, input_manifest=input_manifest), rejected
        except (ValueError, ValidationError) as exc:
            rejected.append({"strategy_name": name, "reason": str(exc)})
    raise ValueError(f"no executable strategy; rejected={rejected}")
