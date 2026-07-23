"""Cost and risk estimation heuristics for ToolUsePlannerAgent.

Deterministic, transparent, testable — NOT machine learning.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autovs.schemas import ActionType, StrictModel


# ═══════════════════════════════════════════════════════════════════════
# Scoring weights — centralized, NOT scattered
# ═══════════════════════════════════════════════════════════════════════

# Scientific importance penalty: required=0, recommended=0.2, optional=0.5
SCIENTIFIC_IMPORTANCE_WEIGHTS = {"required": 0.0, "recommended": 0.2, "optional": 0.5}

# Capability availability penalty
UNAVAILABLE_PENALTY = float("inf")   # forbidden
DEGRADED_PENALTY = 0.3               # still usable but penalized

# GPU constraint penalty
GPU_CONSTRAINT_PENALTY = 0.5

# Format conversion penalty (extra steps)
FORMAT_CONVERSION_PENALTY = 0.1

# Cost normalization: each step relative to baseline docking cost
RELATIVE_COSTS: dict[ActionType, float] = {
    ActionType.INPUT_VALIDATION: 0.05,
    ActionType.TARGET_STRUCTURE_ACQUISITION: 0.2,
    ActionType.POCKET_DEFINITION: 0.15,
    ActionType.PROTEIN_PREPARATION: 0.1,
    ActionType.MOLECULE_STANDARDIZATION: 0.15,
    ActionType.MOLECULE_STANDARDIZATION_V2: 0.15,
    ActionType.CONFORMER_GENERATION: 0.2,
    ActionType.PHYSICOCHEMICAL_FILTERING: 0.1,
    ActionType.IONIZATION_ENUMERATION: 0.1,
    ActionType.LIGAND_3D_ENUMERATION: 0.5,
    ActionType.PDBQT_PARAMETERIZATION: 0.1,
    ActionType.MOLECULAR_DOCKING: 1.0,   # baseline
    ActionType.POSE_EXTRACTION: 0.1,
    ActionType.INTERACTION_ANALYSIS: 0.3,
    ActionType.ADMET_FILTERING: 0.2,
    ActionType.SHORT_MD: 5.0,
    ActionType.MOLECULAR_DYNAMICS: 15.0,
    ActionType.FINAL_RANKING: 0.05,
    ActionType.REPORT_GENERATION: 0.05,
    ActionType.STRUCTURE_ANALYSIS: 0.1,
    ActionType.PROTEIN_REPAIR: 0.1,
    ActionType.PROTONATION: 0.2,
    ActionType.FORMAT_CONVERSION: 0.05,
    ActionType.DIVERSITY_SELECTION: 0.05,
    ActionType.TARGET_STRUCTURE_PREDICTION: 2.0,
}

# Failure risk: probability estimate per action type (~0.0 to ~1.0)
FAILURE_RISKS: dict[ActionType, float] = {
    ActionType.INPUT_VALIDATION: 0.02,
    ActionType.TARGET_STRUCTURE_ACQUISITION: 0.10,
    ActionType.POCKET_DEFINITION: 0.08,
    ActionType.PROTEIN_PREPARATION: 0.05,
    ActionType.MOLECULE_STANDARDIZATION: 0.05,
    ActionType.MOLECULE_STANDARDIZATION_V2: 0.05,
    ActionType.CONFORMER_GENERATION: 0.05,
    ActionType.PHYSICOCHEMICAL_FILTERING: 0.03,
    ActionType.IONIZATION_ENUMERATION: 0.05,
    ActionType.LIGAND_3D_ENUMERATION: 0.10,
    ActionType.PDBQT_PARAMETERIZATION: 0.05,
    ActionType.MOLECULAR_DOCKING: 0.15,
    ActionType.POSE_EXTRACTION: 0.05,
    ActionType.INTERACTION_ANALYSIS: 0.10,
    ActionType.ADMET_FILTERING: 0.05,
    ActionType.SHORT_MD: 0.30,
    ActionType.MOLECULAR_DYNAMICS: 0.40,
    ActionType.FINAL_RANKING: 0.02,
    ActionType.REPORT_GENERATION: 0.01,
    ActionType.STRUCTURE_ANALYSIS: 0.03,
    ActionType.PROTEIN_REPAIR: 0.10,
    ActionType.PROTONATION: 0.08,
    ActionType.FORMAT_CONVERSION: 0.02,
    ActionType.DIVERSITY_SELECTION: 0.03,
    ActionType.TARGET_STRUCTURE_PREDICTION: 0.30,
}

# Walltime estimates in seconds
WALLTIME_ESTIMATES: dict[ActionType, int] = {
    ActionType.INPUT_VALIDATION: 30,
    ActionType.TARGET_STRUCTURE_ACQUISITION: 60,
    ActionType.POCKET_DEFINITION: 120,
    ActionType.PROTEIN_PREPARATION: 60,
    ActionType.MOLECULE_STANDARDIZATION: 60,
    ActionType.MOLECULE_STANDARDIZATION_V2: 60,
    ActionType.CONFORMER_GENERATION: 300,
    ActionType.PHYSICOCHEMICAL_FILTERING: 60,
    ActionType.IONIZATION_ENUMERATION: 120,
    ActionType.LIGAND_3D_ENUMERATION: 600,
    ActionType.PDBQT_PARAMETERIZATION: 120,
    ActionType.MOLECULAR_DOCKING: 3600,
    ActionType.POSE_EXTRACTION: 60,
    ActionType.INTERACTION_ANALYSIS: 1800,
    ActionType.ADMET_FILTERING: 300,
    ActionType.SHORT_MD: 36000,
    ActionType.MOLECULAR_DYNAMICS: 259200,
    ActionType.FINAL_RANKING: 10,
    ActionType.REPORT_GENERATION: 10,
    ActionType.STRUCTURE_ANALYSIS: 30,
    ActionType.PROTEIN_REPAIR: 120,
    ActionType.PROTONATION: 120,
    ActionType.FORMAT_CONVERSION: 30,
    ActionType.DIVERSITY_SELECTION: 30,
    ActionType.TARGET_STRUCTURE_PREDICTION: 7200,
}


# ═══════════════════════════════════════════════════════════════════════
# Estimate models
# ═══════════════════════════════════════════════════════════════════════

class StepCostEstimate(StrictModel):
    """单步成本估计。"""
    relative_cost: float = Field(ge=0.0)
    estimated_walltime_seconds: int | None = None
    estimated_cpu_hours: float | None = None
    estimated_gpu_hours: float | None = None
    confidence: Literal["low", "medium", "high"] = "low"


class StepRiskEstimate(StrictModel):
    """单步失败风险估计。"""
    failure_probability: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "medium", "high", "critical"] = "low"
    reasons: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)


def estimate_step_cost(
    action: ActionType,
    cpus: int = 1,
    gpu_required: bool = False,
) -> StepCostEstimate:
    """估算单步成本。"""
    rel = RELATIVE_COSTS.get(action, 0.5)
    walltime = WALLTIME_ESTIMATES.get(action, 300)
    cpu_hours = (walltime / 3600.0) * cpus if cpus > 0 else None
    gpu_hours = cpu_hours if gpu_required else None
    return StepCostEstimate(
        relative_cost=rel,
        estimated_walltime_seconds=walltime,
        estimated_cpu_hours=cpu_hours,
        estimated_gpu_hours=gpu_hours,
        confidence="low",
    )


def estimate_step_risk(
    action: ActionType,
    availability: Literal["available", "degraded", "unavailable"] = "available",
) -> StepRiskEstimate:
    """估算单步失败风险。"""
    base = FAILURE_RISKS.get(action, 0.1)
    reasons: list[str] = []
    mitigations: list[str] = []

    if availability == "degraded":
        base += 0.15
        reasons.append("capability is degraded")
        mitigations.append("consider alternative tool if available")
    elif availability == "unavailable":
        base = 1.0
        reasons.append("capability is unavailable")
        mitigations.append("must be excluded from plan")

    base = min(base, 1.0)

    if base <= 0.05:
        level: Literal["low", "medium", "high", "critical"] = "low"
    elif base <= 0.15:
        level = "medium"
    elif base < 1.0:
        level = "high"
    else:
        level = "critical"

    return StepRiskEstimate(
        failure_probability=round(base, 4),
        level=level,
        reasons=reasons,
        mitigations=mitigations,
    )


def candidate_score(
    importance: str,
    availability: Literal["available", "degraded", "unavailable"],
    cpu_only: bool,
    gpu_required: bool,
    action: ActionType,
) -> float:
    """计算候选步骤的综合评分（越低越好）。"""
    if availability == "unavailable":
        return UNAVAILABLE_PENALTY

    sci = SCIENTIFIC_IMPORTANCE_WEIGHTS.get(importance, 0.3)
    deg = DEGRADED_PENALTY if availability == "degraded" else 0.0
    cost = RELATIVE_COSTS.get(action, 0.5)
    risk = FAILURE_RISKS.get(action, 0.1)
    gpu = GPU_CONSTRAINT_PENALTY if (cpu_only and gpu_required) else 0.0

    return sci + deg + min(cost, 1.0) + risk + gpu
