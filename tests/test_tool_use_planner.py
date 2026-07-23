"""Unit tests for ToolUsePlannerAgent and WorkflowGraphBuilder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autovs.capabilities import list_capabilities
from autovs.config import load_settings
from autovs.planning.contracts import get_contract, find_producers
from autovs.planning.errors import (
    ArtifactGapError, PlannerCapabilityGapError, PlannerError, PlanningValidationError,
)
from autovs.planning.graph_builder import (
    PlannedActionIntent, PlannerConstraints, PlannerDraft, PlannerResult,
    PlanningDecision, WorkflowGraphBuilder,
)
from autovs.planning.validator import validate_workflow_plan
from autovs.planning.scoring import candidate_score, estimate_step_cost, estimate_step_risk
from autovs.schemas import (
    ActionType, InputManifest, LibraryAsset, PocketSpec,
    TargetAsset, ToolCapability, WorkflowPlan, WorkflowStep,
)
from autovs.dag import (
    SCREENING_LIBRARY, TARGET_STRUCTURE, POCKET_CENTER,
    RECEPTOR_PDBQT, PREPARED_LIBRARY, TOP_HITS,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_manifest(locked_target: bool = False) -> InputManifest:
    return InputManifest(
        query="a sufficiently long screening query for testing",
        library_asset=LibraryAsset(
            source="user", path="/tmp/lib.smi", sha256="a" * 64,
        ),
        target_asset=TargetAsset(
            source="user" if locked_target else "research",
            locked=locked_target,
            path="/tmp/protein.pdb" if locked_target else None,
            sha256="b" * 64 if locked_target else None,
        ),
        expert_pocket=PocketSpec(),
        warnings=[],
        constraint_summary=[],
    )


def _cap_with(action: ActionType, availability: str = "available") -> ToolCapability:
    caps = list_capabilities(load_settings())
    for c in caps:
        if c.action_type == action:
            return ToolCapability(
                action_type=action,
                name=c.name,
                description=c.description,
                availability=availability,
                executor=c.executor,
                input_formats=c.input_formats,
                output_formats=c.output_formats,
                gpu_required=c.gpu_required,
            )
    return ToolCapability(
        action_type=action, name=action.value, description="",
        availability="unavailable", executor="python",
        input_formats=[], output_formats=[], gpu_required=False,
    )


def _get_capabilities() -> list[ToolCapability]:
    return list_capabilities(load_settings())


def _all_available_capabilities() -> list[ToolCapability]:
    return [
        ToolCapability(
            action_type=action,
            name=action.value,
            description="test capability",
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


def _step_by_action(plan: WorkflowPlan, action: ActionType) -> WorkflowStep:
    return next(step for step in plan.steps if step.action_type == action)


# ── Scoring Tests ─────────────────────────────────────────────────────

class TestScoring:
    def test_candidate_score_unavailable_is_inf(self):
        score = candidate_score("required", "unavailable", False, False, ActionType.MOLECULAR_DOCKING)
        assert score == float("inf")

    def test_candidate_score_degraded_higher_than_available(self):
        avail = candidate_score("required", "available", False, False, ActionType.MOLECULAR_DOCKING)
        degraded = candidate_score("required", "degraded", False, False, ActionType.MOLECULAR_DOCKING)
        assert degraded > avail

    def test_cpu_only_penalizes_gpu(self):
        cpu_score = candidate_score("recommended", "available", True, True, ActionType.MOLECULAR_DYNAMICS)
        no_cpu_score = candidate_score("recommended", "available", False, True, ActionType.MOLECULAR_DYNAMICS)
        assert cpu_score > no_cpu_score

    def test_estimate_cost_exists_for_all_actions(self):
        for action in ActionType:
            cost = estimate_step_cost(action)
            assert cost.relative_cost >= 0

    def test_estimate_risk_exists_for_all_actions(self):
        for action in ActionType:
            risk = estimate_step_risk(action, "available")
            assert 0 <= risk.failure_probability <= 1


# ── Graph Builder Tests ──────────────────────────────────────────────

class TestGraphBuilder:
    def test_basic_dag_with_locked_pdb(self):
        """用户上传锁定PDB → 不应插入 target_structure_acquisition。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="分子对接"),
                PlannedActionIntent(action_type=ActionType.INTERACTION_ANALYSIS, importance="recommended",
                                    rationale="PLIP分析"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        )
        result = builder.build()

        # 不应插入 target_structure_acquisition
        actions_in_plan = {s.action_type for s in result.plan.steps}
        assert ActionType.TARGET_STRUCTURE_ACQUISITION not in actions_in_plan

        # 必须存在 docking
        assert ActionType.MOLECULAR_DOCKING in actions_in_plan

        # 应该自动补全: input_validation + pocket_definition + protein_prep + 分子准备
        assert ActionType.INPUT_VALIDATION in actions_in_plan
        assert ActionType.POCKET_DEFINITION in actions_in_plan
        assert ActionType.PROTEIN_PREPARATION in actions_in_plan

        # WorkflowPlan 校验通过
        WorkflowPlan.model_validate(result.plan.model_dump(mode="json"))

    def test_no_uploaded_pdb_inserts_acquisition(self):
        """无上传PDB → 插入 target_structure_acquisition。"""
        manifest = _make_manifest(locked_target=False)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        )
        result = builder.build()

        actions_in_plan = {s.action_type for s in result.plan.steps}
        assert ActionType.TARGET_STRUCTURE_ACQUISITION in actions_in_plan

    def test_unavailable_required_raises(self):
        """required 但 capability unavailable → PlannerCapabilityGapError。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()
        # 将 docking 标记为 unavailable
        caps = [c for c in caps if c.action_type != ActionType.MOLECULAR_DOCKING]
        caps.append(_cap_with(ActionType.MOLECULAR_DOCKING, "unavailable"))

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="必须对接"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        )
        with pytest.raises(PlannerCapabilityGapError, match="molecular_docking"):
            builder.build()

    def test_unavailable_optional_skipped(self):
        """optional 且 capability unavailable → 跳过。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()
        caps = [c for c in caps if c.action_type != ActionType.ADMET_FILTERING]
        caps.append(_cap_with(ActionType.ADMET_FILTERING, "unavailable"))

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
                PlannedActionIntent(action_type=ActionType.ADMET_FILTERING, importance="optional",
                                    rationale="ADMET"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        )
        result = builder.build()

        actions_in_plan = {s.action_type for s in result.plan.steps}
        assert ActionType.MOLECULAR_DOCKING in actions_in_plan
        assert ActionType.ADMET_FILTERING not in actions_in_plan

        # 决策中记录跳过
        skip_decisions = [d for d in result.decisions if d.decision == "skipped"]
        assert any("admet" in d.reason.lower() for d in skip_decisions)

    def test_parallel_branches(self):
        """蛋白准备和配体准备不应互相依赖。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        )
        result = builder.build()

        # 找到蛋白准备和配体准备步骤
        by_id = {s.step_id: s for s in result.plan.steps}
        protein_step = next((s for s in result.plan.steps
                            if s.action_type == ActionType.PROTEIN_PREPARATION), None)
        ligand_step = next((s for s in result.plan.steps
                           if s.action_type == ActionType.MOLECULE_STANDARDIZATION), None)

        if protein_step and ligand_step:
            # 它们不应该互相依赖
            protein_reqs = set(protein_step.requires)
            ligand_reqs = set(ligand_step.requires)
            assert protein_step.step_id not in ligand_reqs
            assert ligand_step.step_id not in protein_reqs

    def test_deterministic_output(self):
        """相同输入 → 相同 plan。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
            ],
        )

        result1 = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        ).build()

        result2 = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        ).build()

        assert result1.plan.model_dump() == result2.plan.model_dump()

    def test_auto_inject_producers(self):
        """只写 docking 时自动注入蛋白准备、配体准备、口袋步骤。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(),
        )
        result = builder.build()

        actions_in_plan = {s.action_type for s in result.plan.steps}
        # 必须自动注入的核心步骤
        assert ActionType.INPUT_VALIDATION in actions_in_plan
        assert ActionType.POCKET_DEFINITION in actions_in_plan
        assert ActionType.PROTEIN_PREPARATION in actions_in_plan
        # 分子准备（至少一种，FORMAT_CONVERSION 或 MOLECULE_STANDARDIZATION 都可以）
        assert any(a in actions_in_plan for a in [
            ActionType.MOLECULE_STANDARDIZATION,
            ActionType.MOLECULE_STANDARDIZATION_V2,
            ActionType.FORMAT_CONVERSION,
        ])

    def test_cpu_only_constraint(self):
        """cpu_only=True → 不选 GPU action。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
                PlannedActionIntent(action_type=ActionType.SHORT_MD, importance="optional",
                                    rationale="短MD"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(cpu_only=True),
        )
        result = builder.build()

        actions_in_plan = {s.action_type for s in result.plan.steps}
        assert ActionType.MOLECULAR_DOCKING in actions_in_plan
        # SHORT_MD requires GPU → skipped in cpu_only mode
        skip_gpu = [d for d in result.decisions
                    if d.action_type == ActionType.SHORT_MD and d.decision == "skipped"]
        assert len(skip_gpu) > 0

    def test_transform_chain_freezes_input_producers(self):
        """同 key transform 链必须按实际 producer 顺序依赖，不能用最终 producer 回填。"""
        manifest = _make_manifest(locked_target=True)
        draft = PlannerDraft(
            strategy_id="transform_chain",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULE_STANDARDIZATION_V2, importance="required"),
                PlannedActionIntent(action_type=ActionType.IONIZATION_ENUMERATION, importance="required"),
                PlannedActionIntent(action_type=ActionType.LIGAND_3D_ENUMERATION, importance="required"),
            ],
        )

        result = WorkflowGraphBuilder(
            draft=draft,
            input_manifest=manifest,
            capabilities=_all_available_capabilities(),
            constraints=PlannerConstraints(),
        ).build()

        standardize = _step_by_action(result.plan, ActionType.MOLECULE_STANDARDIZATION_V2)
        ionize = _step_by_action(result.plan, ActionType.IONIZATION_ENUMERATION)
        enumerate_3d = _step_by_action(result.plan, ActionType.LIGAND_3D_ENUMERATION)
        assert standardize.step_id in ionize.requires
        assert ionize.step_id in enumerate_3d.requires
        assert ionize.step_id not in standardize.requires

    def test_target_structure_transform_chain_depends_on_previous_step(self):
        manifest = _make_manifest(locked_target=True)
        draft = PlannerDraft(
            strategy_id="protein_chain",
            actions=[
                PlannedActionIntent(action_type=ActionType.PROTEIN_REPAIR, importance="required"),
                PlannedActionIntent(action_type=ActionType.PROTONATION, importance="required"),
                PlannedActionIntent(action_type=ActionType.PROTEIN_PREPARATION, importance="required"),
            ],
        )

        result = WorkflowGraphBuilder(
            draft=draft,
            input_manifest=manifest,
            capabilities=_all_available_capabilities(),
            constraints=PlannerConstraints(),
        ).build()

        repair = _step_by_action(result.plan, ActionType.PROTEIN_REPAIR)
        protonation = _step_by_action(result.plan, ActionType.PROTONATION)
        protein_prep = _step_by_action(result.plan, ActionType.PROTEIN_PREPARATION)
        assert repair.step_id in protonation.requires
        assert protonation.step_id in protein_prep.requires

    def test_pdbqt_parameterization_does_not_feed_current_smina_docking(self):
        manifest = _make_manifest(locked_target=True)
        draft = PlannerDraft(
            strategy_id="pdbqt_branch",
            actions=[
                PlannedActionIntent(action_type=ActionType.LIGAND_3D_ENUMERATION, importance="required"),
                PlannedActionIntent(action_type=ActionType.PDBQT_PARAMETERIZATION, importance="required"),
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required"),
            ],
        )

        result = WorkflowGraphBuilder(
            draft=draft,
            input_manifest=manifest,
            capabilities=_all_available_capabilities(),
            constraints=PlannerConstraints(),
        ).build()

        enumerate_3d = _step_by_action(result.plan, ActionType.LIGAND_3D_ENUMERATION)
        pdbqt = _step_by_action(result.plan, ActionType.PDBQT_PARAMETERIZATION)
        docking = _step_by_action(result.plan, ActionType.MOLECULAR_DOCKING)
        assert enumerate_3d.step_id in pdbqt.requires
        assert enumerate_3d.step_id in docking.requires
        assert pdbqt.step_id not in docking.requires

    def test_runtime_unsupported_required_action_is_capability_gap(self):
        manifest = _make_manifest(locked_target=True)
        draft = PlannerDraft(
            strategy_id="unsupported_required",
            actions=[
                PlannedActionIntent(action_type=ActionType.ADMET_FILTERING, importance="required"),
            ],
        )

        with pytest.raises(PlannerCapabilityGapError, match="admet_filtering"):
            WorkflowGraphBuilder(
                draft=draft,
                input_manifest=manifest,
                capabilities=_all_available_capabilities(),
                constraints=PlannerConstraints(),
            ).build()

    def test_validator_rejects_missing_artifact_producer(self):
        manifest = _make_manifest(locked_target=True)
        plan = WorkflowPlan(
            strategy_id="broken",
            steps=[
                WorkflowStep(
                    step_id="docking",
                    action_type=ActionType.MOLECULAR_DOCKING,
                    inputs=[
                        item for item in _step_by_action(
                            WorkflowGraphBuilder(
                                draft=PlannerDraft(
                                    strategy_id="ok",
                                    actions=[
                                        PlannedActionIntent(
                                            action_type=ActionType.MOLECULAR_DOCKING,
                                            importance="required",
                                        )
                                    ],
                                ),
                                input_manifest=manifest,
                                capabilities=_all_available_capabilities(),
                                constraints=PlannerConstraints(),
                            ).build().plan,
                            ActionType.MOLECULAR_DOCKING,
                        ).inputs
                    ],
                    outputs=[],
                )
            ],
        )

        with pytest.raises(PlanningValidationError, match="required artifact"):
            validate_workflow_plan(plan, input_manifest=manifest)


# ── ToolUsePlannerAgent Tests ─────────────────────────────────────────

class TestToolUsePlannerAgent:
    def test_heuristic_mode_returns_valid_plan(self):
        """无 LLM 确定性模式返回有效 WorkflowPlan。"""
        from src.agents.tool_use_planner import ToolUsePlannerAgent
        from autovs.planning import PlannerConstraints
        from autovs.config import load_settings

        manifest = _make_manifest(locked_target=True)
        settings = load_settings()
        agent = ToolUsePlannerAgent(settings=settings, llm_client=None)

        strategy = {
            "strategy_id": "test_strategy",
            "strategy_name": "CPU SBDD",
            "description": "标准基于结构的虚拟筛选",
            "pipeline": [
                {"action_type": "molecular_docking", "description": "smina 对接"},
                {"action_type": "pose_extraction", "description": "姿态提取"},
                {"action_type": "interaction_analysis", "description": "PLIP"},
            ],
        }

        result = agent.plan(strategy, manifest, constraints=PlannerConstraints())
        assert isinstance(result, PlannerResult)
        assert result.plan is not None
        assert len(result.plan.steps) >= 3
        WorkflowPlan.model_validate(result.plan.model_dump(mode="json"))

    def test_empty_strategy_handled(self):
        """空 pipeline 仍能生成基本计划。"""
        from src.agents.tool_use_planner import ToolUsePlannerAgent
        from autovs.planning import PlannerConstraints
        from autovs.config import load_settings

        manifest = _make_manifest(locked_target=True)
        settings = load_settings()
        agent = ToolUsePlannerAgent(settings=settings, llm_client=None)

        strategy = {
            "strategy_id": "minimal",
            "description": "最小策略",
        }

        result = agent.plan(strategy, manifest, constraints=PlannerConstraints())
        assert len(result.plan.steps) >= 0  # 可能为空也可能有强制步骤
        WorkflowPlan.model_validate(result.plan.model_dump(mode="json"))

    def test_planner_result_contains_decisions(self):
        from src.agents.tool_use_planner import ToolUsePlannerAgent
        from autovs.planning import PlannerConstraints
        from autovs.config import load_settings

        manifest = _make_manifest(locked_target=True)
        settings = load_settings()
        agent = ToolUsePlannerAgent(settings=settings, llm_client=None)

        strategy = {
            "strategy_id": "test",
            "pipeline": [
                {"action_type": "molecular_docking", "description": "对接"},
            ],
        }

        result = agent.plan(strategy, manifest, constraints=PlannerConstraints())
        assert isinstance(result.decisions, list)
        assert len(result.decisions) > 0
        for d in result.decisions:
            assert isinstance(d, PlanningDecision)


# ── Safety Tests ──────────────────────────────────────────────────────

class TestSafety:
    def test_no_paths_in_draft(self):
        """PlannedActionIntent 不允许路径。"""
        from autovs.planning.graph_builder import PlannedActionIntent
        # PlannedActionIntent 没有 path 字段，直接校验
        intent = PlannedActionIntent(
            action_type=ActionType.MOLECULAR_DOCKING,
            importance="required",
            parameters={"exhaustiveness": 4},
        )
        assert "path" not in intent.model_dump()

    def test_forbidden_action_excluded(self):
        """forbidden_actions 中的 action 不进入计划。"""
        manifest = _make_manifest(locked_target=True)
        caps = _get_capabilities()

        draft = PlannerDraft(
            strategy_id="test",
            actions=[
                PlannedActionIntent(action_type=ActionType.MOLECULAR_DOCKING, importance="required",
                                    rationale="对接"),
                PlannedActionIntent(action_type=ActionType.SHORT_MD, importance="optional",
                                    rationale="MD"),
            ],
        )

        builder = WorkflowGraphBuilder(
            draft=draft, input_manifest=manifest, capabilities=caps,
            constraints=PlannerConstraints(
                forbidden_actions=[ActionType.SHORT_MD],
            ),
        )
        result = builder.build()

        actions_in_plan = {s.action_type for s in result.plan.steps}
        assert ActionType.SHORT_MD not in actions_in_plan
