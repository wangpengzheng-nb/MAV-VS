"""Deterministic WorkflowGraphBuilder.

Builds a non-linear DAG from PlannerDraft + ActionContracts + ArtifactRegistry.
NO LLM involvement in dependency creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from autovs.schemas import (
    ActionType, ArtifactRef, InputManifest, ResourceProfile,
    StrictModel, WorkflowPlan, WorkflowStep,
)
from autovs.planning.contracts import (
    ACTION_CONTRACTS, ARTIFACT_REGISTRY, ActionIOContract,
    find_producers, get_contract,
)
from autovs.planning.errors import (
    ArtifactGapError, AssetLockViolation, PlannerCapabilityGapError,
    PlanningValidationError,
)
from autovs.planning.scoring import (
    DEGRADED_PENALTY, candidate_score, estimate_step_cost, estimate_step_risk,
)
from autovs.capabilities import list_capabilities
from autovs.config import Settings


# ═══════════════════════════════════════════════════════════════════════
# Planner internal models
# ═══════════════════════════════════════════════════════════════════════

class PlannedActionIntent(StrictModel):
    action_type: ActionType
    importance: Literal["required", "recommended", "optional"] = "required"
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    preferred_executor: str | None = None


class PlannerDraft(StrictModel):
    strategy_id: str
    actions: list[PlannedActionIntent] = Field(default_factory=list)
    scientific_objectives: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PlanningDecision(StrictModel):
    step_id: str
    action_type: ActionType
    decision: Literal["added", "skipped", "replaced", "converted"]
    reason: str


class PlanningAlternative(StrictModel):
    action_type: ActionType
    preferred: str
    alternatives: list[str] = Field(default_factory=list)
    chosen: str


class PlannerConstraints(StrictModel):
    cpu_only: bool = False
    allow_gpu: bool = True
    allow_degraded_capabilities: bool = True
    max_steps: int = Field(default=30, ge=1, le=100)
    max_walltime_seconds: int | None = None
    max_cpu_hours: float | None = None
    max_gpu_hours: float | None = None
    max_relative_cost: float | None = None
    max_failure_risk: float | None = None
    required_actions: list[ActionType] = Field(default_factory=list)
    forbidden_actions: list[ActionType] = Field(default_factory=list)


class PlannerResult(StrictModel):
    plan: WorkflowPlan
    decisions: list[PlanningDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    capability_gaps: list[str] = Field(default_factory=list)
    alternatives_considered: list[PlanningAlternative] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════════════

class WorkflowGraphBuilder:
    """确定性 DAG 构建器。

    根据 PlannerDraft、ActionContracts 和 ArtifactRegistry，
    构建一个非线性的 WorkflowPlan。
    """

    def __init__(
        self,
        draft: PlannerDraft,
        input_manifest: InputManifest,
        capabilities: list[Any],  # list[ToolCapability]
        constraints: PlannerConstraints,
        settings: Settings | None = None,
    ):
        self.draft = draft
        self.manifest = input_manifest
        self.constraints = constraints
        self.settings = settings

        # 构建快捷查找表
        self._cap_map: dict[ActionType, Any] = {}
        for c in capabilities:
            self._cap_map[c.action_type] = c

        # 内部状态
        self._available_artifacts: dict[str, str] = {}  # key → producer_step_id
        self._steps: dict[str, WorkflowStep] = {}       # step_id → WorkflowStep
        self._decisions: list[PlanningDecision] = []
        self._warnings: list[str] = list(draft.warnings)
        self._gaps: list[str] = []
        self._alternatives: list[PlanningAlternative] = []
        self._step_counter: dict[str, int] = {}
        self._visited_actions: set[ActionType] = set()   # 防止递归注入

    # ── 公共入口 ──────────────────────────────────────────────────

    def build(self) -> PlannerResult:
        # 1. 初始化已有 artifact
        self._init_available_artifacts()

        # 2. 处理 draft actions
        required = [a for a in self.draft.actions if a.importance == "required"]
        recommended = [a for a in self.draft.actions if a.importance == "recommended"]
        optional = [a for a in self.draft.actions if a.importance == "optional"]

        # required 优先
        for intent in required:
            self._add_action(intent, must_succeed=True)

        # recommended
        for intent in recommended:
            self._add_action(intent, must_succeed=False)

        # optional
        for intent in optional:
            self._add_action(intent, must_succeed=False)

        # 3. 强制步骤
        self._ensure_mandatory_steps()

        # 4. 拓扑排序 + 构建依赖
        ordered = self._topological_sort()
        self._build_requires(ordered)

        # 5. 生成 WorkflowPlan
        steps = [self._steps[sid] for sid in ordered]
        plan = WorkflowPlan(
            strategy_id=self.draft.strategy_id,
            steps=steps,
        )

        return PlannerResult(
            plan=plan,
            decisions=self._decisions,
            warnings=self._warnings,
            capability_gaps=self._gaps,
            alternatives_considered=self._alternatives,
        )

    # ── 内部方法 ──────────────────────────────────────────────────

    def _unique_step_id(self, action: ActionType) -> str:
        base = action.value.replace("_", "-")
        if base not in self._step_counter:
            self._step_counter[base] = 0
            return base
        self._step_counter[base] += 1
        return f"{base}-{self._step_counter[base]}"

    def _init_available_artifacts(self) -> None:
        """从 InputManifest 初始化已有 artifact。"""
        from autovs.dag import SCREENING_LIBRARY, TARGET_STRUCTURE

        # screening_library 始终可用
        self._available_artifacts[SCREENING_LIBRARY] = "_manifest"

        # target_structure: 用户上传时可用
        if self.manifest.target_asset.locked and self.manifest.target_asset.path:
            self._available_artifacts[TARGET_STRUCTURE] = "_manifest"

        # research evidence
        self._available_artifacts["_research_path"] = "_manifest"

    def _capacity(self, action: ActionType) -> Literal["available", "degraded", "unavailable"]:
        cap = self._cap_map.get(action)
        if cap is None:
            return "unavailable"
        return cap.availability

    def _add_action(self, intent: PlannedActionIntent, must_succeed: bool) -> None:
        """将一个 PlannedActionIntent 加入图中。"""
        action = intent.action_type
        contract = get_contract(action)

        # 检查能力
        availability = self._capacity(action)
        if availability == "unavailable":
            if must_succeed:
                cap = self._cap_map.get(action)
                reason = cap.reason if cap else "未注册"
                raise PlannerCapabilityGapError(action.value, reason)
            else:
                self._decisions.append(PlanningDecision(
                    step_id="",
                    action_type=action,
                    decision="skipped",
                    reason=f"{action.value} capability unavailable",
                ))
                return

        if availability == "degraded" and not self.constraints.allow_degraded_capabilities:
            if must_succeed:
                raise PlannerCapabilityGapError(
                    action.value,
                    "degraded capability not allowed by constraints",
                )
            else:
                self._decisions.append(PlanningDecision(
                    step_id="",
                    action_type=action,
                    decision="skipped",
                    reason=f"{action.value} capability is degraded and constraints forbid",
                ))
                return

        # GPU 约束
        cap = self._cap_map.get(action)
        gpu = cap.gpu_required if cap else False
        if self.constraints.cpu_only and gpu:
            if must_succeed:
                raise PlannerCapabilityGapError(
                    action.value, "GPU required but cpu_only=True",
                )
            else:
                self._decisions.append(PlanningDecision(
                    step_id="",
                    action_type=action,
                    decision="skipped",
                    reason=f"{action.value} requires GPU but cpu_only=True",
                ))
                return

        # 检查 forbidden
        if action in self.constraints.forbidden_actions:
            self._decisions.append(PlanningDecision(
                step_id="",
                action_type=action,
                decision="skipped",
                reason=f"{action.value} is forbidden by constraints",
            ))
            return

        # 补齐输入
        self._ensure_inputs(action, contract)

        # 添加步骤
        sid = self._unique_step_id(action)
        step = WorkflowStep(
            step_id=sid,
            action_type=action,
            parameters=intent.parameters,
            resource_profile=ResourceProfile(
                executor="python",
                gpu_required=gpu,
            ),
        )
        self._steps[sid] = step

        # 注册输出
        if contract:
            for output_key in contract.outputs:
                self._available_artifacts[output_key] = sid

        self._decisions.append(PlanningDecision(
            step_id=sid,
            action_type=action,
            decision="added",
            reason=intent.rationale or f"scientific: {contract.scientific_role if contract else 'unknown'}",
        ))

        if availability == "degraded":
            self._warnings.append(f"{action.value} 使用了 degraded 能力")

    def _ensure_inputs(self, action: ActionType, contract: ActionIOContract | None) -> None:
        """确保 action 的所有 required input 都有来源。"""
        if contract is None:
            return
        for req_key in contract.required_inputs:
            if req_key in self._available_artifacts:
                continue
            # 查找 producer
            producers = find_producers(req_key)
            # 优先 source producer（从无到有创建），其次 transform producer
            source_producers = [p for p in producers
                                if get_contract(p) and get_contract(p).is_source_producer]
            available_source = [p for p in source_producers
                                if self._capacity(p) != "unavailable"
                                and p not in self.constraints.forbidden_actions]
            if available_source:
                # 优先 source
                available_producers = available_source
            else:
                available_producers = [p for p in producers
                                       if self._capacity(p) != "unavailable"
                                       and p not in self.constraints.forbidden_actions]
            if not available_producers:
                raise ArtifactGapError(req_key, action.value)

            # 选最佳 producer（最低评分）
            best = min(
                available_producers,
                key=lambda p: candidate_score(
                    "required",
                    self._capacity(p),
                    self.constraints.cpu_only,
                    self._cap_map.get(p).gpu_required if self._cap_map.get(p) else False,
                    p,
                ),
            )
            self._inject_producer(best, req_key)

    def _inject_producer(self, action: ActionType, needed_artifact: str) -> None:
        """递归注入 producer 步骤。"""
        # 防止重复注入或自循环
        if action in self._visited_actions:
            return
        self._visited_actions.add(action)

        if action in self._steps:
            return  # 已存在

        contract = get_contract(action)
        if contract is None:
            raise ArtifactGapError(needed_artifact, action.value)

        # 递归补齐 producer 的输入
        for req_key in contract.required_inputs:
            if req_key not in self._available_artifacts:
                producers = find_producers(req_key)
                source_producers = [p for p in producers
                                    if get_contract(p) and get_contract(p).is_source_producer]
                avail_source = [p for p in source_producers
                                if self._capacity(p) != "unavailable"
                                and p not in self.constraints.forbidden_actions]
                if avail_source:
                    avail = avail_source
                else:
                    avail = [p for p in producers
                             if self._capacity(p) != "unavailable"
                             and p not in self.constraints.forbidden_actions]
                if avail:
                    best = min(
                        avail,
                        key=lambda p: candidate_score(
                            "required",
                            self._capacity(p),
                            self.constraints.cpu_only,
                            self._cap_map.get(p).gpu_required if self._cap_map.get(p) else False,
                            p,
                        ),
                    )
                    self._inject_producer(best, req_key)

        # 添加步骤
        sid = self._unique_step_id(action)
        step = WorkflowStep(
            step_id=sid,
            action_type=action,
            resource_profile=ResourceProfile(executor="python"),
        )
        self._steps[sid] = step

        # 注册输出
        for output_key in contract.outputs:
            self._available_artifacts[output_key] = sid

        self._decisions.append(PlanningDecision(
            step_id=sid,
            action_type=action,
            decision="added",
            reason=f"自动注入以提供 {needed_artifact} 给下游步骤",
        ))

    def _ensure_mandatory_steps(self) -> None:
        """确保强制步骤存在。"""
        from autovs.dag import SCREENING_LIBRARY, TARGET_STRUCTURE

        # input_validation
        if ActionType.INPUT_VALIDATION not in {
            self._steps[s].action_type for s in self._steps
        }:
            self._inject_producer(ActionType.INPUT_VALIDATION, SCREENING_LIBRARY)

        # target_structure_acquisition（需要时）
        if TARGET_STRUCTURE not in self._available_artifacts:
            if self._capacity(ActionType.TARGET_STRUCTURE_ACQUISITION) != "unavailable":
                self._inject_producer(
                    ActionType.TARGET_STRUCTURE_ACQUISITION, TARGET_STRUCTURE,
                )

        # final_ranking
        from autovs.dag import TOP_HITS
        have_ranking = any(
            self._steps[s].action_type == ActionType.FINAL_RANKING
            for s in self._steps
        ) or TOP_HITS in self._available_artifacts
        if not have_ranking:
            from autovs.dag import SCORES_CSV as S
            # 如果还没有 scores_csv，先跳过（纯结构分析任务不需要）
            pass

    def _topological_sort(self) -> list[str]:
        """稳定拓扑排序。"""
        # 构建依赖图
        in_degree: dict[str, int] = {sid: 0 for sid in self._steps}
        deps: dict[str, set[str]] = {sid: set() for sid in self._steps}

        for sid, step in self._steps.items():
            contract = get_contract(step.action_type)
            if contract is None:
                continue
            for req_key in contract.required_inputs:
                producer = self._available_artifacts.get(req_key, "")
                if producer and producer in self._steps and producer != sid:
                    deps[sid].add(producer)

        # 计算入度
        for sid in deps:
            for dep in deps[sid]:
                in_degree[sid] += 1

        # Kahn's algorithm
        import heapq
        heap = [(0, sid) for sid, d in in_degree.items() if d == 0]
        heapq.heapify(heap)
        result = []
        while heap:
            _, sid = heapq.heappop(heap)
            result.append(sid)
            for other in self._steps:
                if sid in deps.get(other, set()):
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        heapq.heappush(heap, (0, other))

        if len(result) != len(self._steps):
            raise PlanningValidationError("DAG contains cycle")

        return result

    def _build_requires(self, ordered: list[str]) -> None:
        """根据拓扑排序后的 artifact 依赖关系填充 requires。"""
        for sid in ordered:
            step = self._steps[sid]
            contract = get_contract(step.action_type)
            if contract is None:
                continue
            deps = set()
            for req_key in contract.required_inputs:
                producer = self._available_artifacts.get(req_key, "")
                if producer and producer in self._steps and producer != sid:
                    deps.add(producer)
            step.requires = sorted(deps)
