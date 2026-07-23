"""WorkflowPlan compile-time validation.

This validator checks the executable DAG contract that Pydantic cannot see:
artifact availability/provenance, action I/O contracts, and DAG executor
resolver/binder coverage.
"""

from __future__ import annotations

from typing import Any

from autovs.dag import (
    INPUT_RESOLVERS, OUTPUT_BINDERS, SCREENING_LIBRARY, TARGET_STRUCTURE,
)
from autovs.planning.contracts import get_artifact, get_contract
from autovs.planning.errors import PlanningValidationError
from autovs.schemas import ActionType, InputManifest, ToolCapability, WorkflowPlan


def _seed_manifest_artifacts(manifest: InputManifest | None) -> dict[str, str]:
    producers = {SCREENING_LIBRARY: "_manifest"}
    if manifest and manifest.target_asset.locked and manifest.target_asset.path:
        producers[TARGET_STRUCTURE] = "_manifest"
    return producers


def _capability_map(capabilities: list[ToolCapability] | None) -> dict[ActionType, ToolCapability]:
    return {cap.action_type: cap for cap in capabilities or []}


def validate_workflow_plan(
    plan: WorkflowPlan,
    *,
    input_manifest: InputManifest | None = None,
    capabilities: list[ToolCapability] | None = None,
    require_runtime_bindings: bool = True,
) -> None:
    """Validate that a WorkflowPlan is executable by the current DAG contract."""

    errors: list[str] = []
    known_steps: set[str] = set()
    producers = _seed_manifest_artifacts(input_manifest)
    caps = _capability_map(capabilities)

    for step in plan.steps:
        if step.step_id in known_steps:
            errors.append(f"{step.step_id}: duplicate step_id")
            continue

        contract = get_contract(step.action_type)
        if contract is None:
            errors.append(f"{step.step_id}: missing action contract for {step.action_type.value}")
            known_steps.add(step.step_id)
            continue

        cap = caps.get(step.action_type)
        if cap and cap.availability == "unavailable":
            errors.append(f"{step.step_id}: capability unavailable: {cap.reason}")

        if require_runtime_bindings and step.action_type != ActionType.REPORT_GENERATION:
            if step.action_type not in INPUT_RESOLVERS:
                errors.append(f"{step.step_id}: missing DAG input resolver")
            if contract.outputs and step.action_type not in OUTPUT_BINDERS:
                errors.append(f"{step.step_id}: missing DAG output binder")

        actual_inputs = {item.name for item in step.inputs}
        actual_outputs = {item.name for item in step.outputs}
        required_inputs = set(contract.required_inputs)
        expected_outputs = set(contract.outputs)

        if not required_inputs.issubset(actual_inputs):
            errors.append(
                f"{step.step_id}: inputs missing contract keys "
                f"{sorted(required_inputs - actual_inputs)}"
            )
        if not expected_outputs.issubset(actual_outputs):
            errors.append(
                f"{step.step_id}: outputs missing contract keys "
                f"{sorted(expected_outputs - actual_outputs)}"
            )

        for ref in [*step.inputs, *step.outputs]:
            if get_artifact(ref.name) is None:
                errors.append(f"{step.step_id}: unknown artifact key {ref.name}")

        for key in contract.required_inputs:
            producer = producers.get(key)
            if producer is None:
                errors.append(f"{step.step_id}: required artifact {key} has no prior producer")
            elif producer != "_manifest" and producer not in step.requires:
                errors.append(
                    f"{step.step_id}: requires missing producer {producer} for artifact {key}"
                )

        for dep in step.requires:
            if dep not in known_steps:
                errors.append(f"{step.step_id}: depends on unknown or later step {dep}")

        known_steps.add(step.step_id)
        for output_key in contract.outputs:
            producers[output_key] = step.step_id

    if errors:
        raise PlanningValidationError("; ".join(errors))


def validate_planner_result(result: Any, **kwargs: Any) -> None:
    """Small convenience wrapper for tests and callers with PlannerResult-like objects."""

    validate_workflow_plan(result.plan, **kwargs)
