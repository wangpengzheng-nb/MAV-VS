"""Deterministic WorkflowGraphBuilder.

Builds a non-linear DAG from PlannerDraft + ActionContracts + ArtifactRegistry.
NO LLM involvement in dependency creation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from autovs.schemas import (
    ActionType, ArtifactRef, InputManifest, ResourceProfile,
    StrictModel, WorkflowPlan, WorkflowStep,
)
from autovs.planning.contracts import (
    ActionIOContract,
    find_producers, get_contract,
)
from autovs.planning.errors import (
    ArtifactGapError, PlannerCapabilityGapError,
    PlanningValidationError,
)
from autovs.planning.scoring import (
    candidate_score, estimate_step_cost, estimate_step_risk,
)
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
        self._step_dependencies: dict[str, set[str]] = {}
        self._decisions: list[PlanningDecision] = []
        self._warnings: list[str] = list(draft.warnings)
        self._gaps: list[str] = []
        self._alternatives: list[PlanningAlternative] = []
        self._step_counter: dict[str, int] = {}
        self._injection_stack: set[ActionType] = set()

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

        # 3. 约束要求的步骤
        for action in self.constraints.required_actions:
            if action not in {step.action_type for step in self._steps.values()}:
                self._add_action(
                    PlannedActionIntent(
                        action_type=action,
                        importance="required",
                        rationale="required by planner constraints",
                    ),
                    must_succeed=True,
                )

        # 4. 强制步骤
        self._ensure_mandatory_steps()

        # 5. 拓扑排序 + 构建依赖
        ordered = self._topological_sort()
        self._build_requires(ordered)
        self._validate_constraints(ordered)

        # 6. 生成 WorkflowPlan
        steps = [self._steps[sid] for sid in ordered]
        plan = WorkflowPlan(
            strategy_id=self.draft.strategy_id,
            steps=steps,
        )
        from autovs.planning.validator import validate_workflow_plan
        validate_workflow_plan(plan, input_manifest=self.manifest, capabilities=list(self._cap_map.values()))

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
        if not self._runtime_supported(action):
            return "unavailable"
        return cap.availability

    def _capability_reason(self, action: ActionType) -> str:
        cap = self._cap_map.get(action)
        if cap is None:
            return "未注册"
        if not self._runtime_supported(action):
            return "DAG executor has no resolver/binder for this action"
        return cap.reason or ""

    def _runtime_supported(self, action: ActionType) -> bool:
        if action == ActionType.REPORT_GENERATION:
            return True
        contract = get_contract(action)
        try:
            from autovs.dag import INPUT_RESOLVERS, OUTPUT_BINDERS
        except Exception:
            return True
        if action not in INPUT_RESOLVERS:
            return False
        if contract and contract.outputs and action not in OUTPUT_BINDERS:
            return False
        return True

    def _add_action(self, intent: PlannedActionIntent, must_succeed: bool) -> None:
        """将一个 PlannedActionIntent 加入图中。"""
        action = intent.action_type
        contract = get_contract(action)

        # 检查能力
        availability = self._capacity(action)
        if availability == "unavailable":
            gap = f"{action.value}: capability unavailable"
            if must_succeed:
                reason = self._capability_reason(action)
                self._gaps.append(f"{action.value}: {reason}")
                raise PlannerCapabilityGapError(action.value, reason)
            else:
                self._gaps.append(gap)
                self._decisions.append(PlanningDecision(
                    step_id="",
                    action_type=action,
                    decision="skipped",
                    reason=f"{action.value} capability unavailable",
                ))
                return

        if availability == "degraded" and not self.constraints.allow_degraded_capabilities:
            if must_succeed:
                self._gaps.append(f"{action.value}: degraded capability not allowed")
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
                self._gaps.append(f"{action.value}: GPU required but cpu_only=True")
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
            if must_succeed:
                raise PlanningValidationError(f"{action.value} is required but forbidden by constraints")
            self._decisions.append(PlanningDecision(
                step_id="",
                action_type=action,
                decision="skipped",
                reason=f"{action.value} is forbidden by constraints",
            ))
            return

        sid = self._add_step(action, contract, parameters=intent.parameters, gpu_required=gpu)

        self._decisions.append(PlanningDecision(
            step_id=sid,
            action_type=action,
            decision="added",
            reason=intent.rationale or f"scientific: {contract.scientific_role if contract else 'unknown'}",
        ))

        if availability == "degraded":
            self._warnings.append(f"{action.value} 使用了 degraded 能力")

    def _artifact_ref(self, key: str) -> ArtifactRef:
        from autovs.planning.contracts import get_artifact

        schema = get_artifact(key)
        fmt = schema.allowed_formats[0] if schema and schema.allowed_formats else "unknown"
        return ArtifactRef(name=key, format=fmt)

    def _resource_profile(self, action: ActionType, contract: ActionIOContract | None, *,
                          gpu_required: bool | None = None) -> ResourceProfile:
        cap = self._cap_map.get(action)
        gpu = bool(gpu_required if gpu_required is not None else (
            cap.gpu_required if cap else (contract.gpu_required if contract else False)
        ))
        executor = contract.executor_type if contract else (cap.executor if cap else "python")
        environment = None
        if action == ActionType.MOLECULAR_DOCKING:
            environment = "smina_stage2"
        elif action == ActionType.INTERACTION_ANALYSIS:
            environment = "plip"
        elif action == ActionType.ADMET_FILTERING:
            environment = "autovs-admet"
        return ResourceProfile(executor=executor, environment=environment, gpu_required=gpu)

    def _ensure_inputs(self, action: ActionType, contract: ActionIOContract | None) -> tuple[list[ArtifactRef], set[str]]:
        """确保 action 的所有 required input 都有来源。"""
        inputs: list[ArtifactRef] = []
        deps: set[str] = set()
        if contract is None:
            return inputs, deps
        for req_key in contract.required_inputs:
            if req_key not in self._available_artifacts:
                # 查找 producer
                producers = find_producers(req_key)
                # 优先 source producer（从无到有创建），其次 transform producer
                source_producers = [p for p in producers
                                    if get_contract(p) and get_contract(p).is_source_producer]
                available_source = [p for p in source_producers
                                    if self._capacity(p) != "unavailable"
                                    and p not in self.constraints.forbidden_actions]
                if available_source:
                    available_producers = available_source
                else:
                    available_producers = [p for p in producers
                                           if self._capacity(p) != "unavailable"
                                           and p not in self.constraints.forbidden_actions]
                if not available_producers:
                    self._gaps.append(f"{req_key}: required by {action.value}")
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
            producer = self._available_artifacts.get(req_key, "")
            if producer and producer in self._steps:
                deps.add(producer)
            inputs.append(self._artifact_ref(req_key))
        for opt_key in contract.optional_inputs:
            if opt_key in self._available_artifacts:
                producer = self._available_artifacts.get(opt_key, "")
                if producer and producer in self._steps:
                    deps.add(producer)
                inputs.append(self._artifact_ref(opt_key))
        return inputs, deps

    def _inject_producer(self, action: ActionType, needed_artifact: str) -> None:
        """递归注入 producer 步骤。"""
        if needed_artifact in self._available_artifacts:
            return
        if action in self._injection_stack:
            raise PlanningValidationError(f"recursive producer injection for {action.value}")
        self._injection_stack.add(action)

        contract = get_contract(action)
        if contract is None:
            self._injection_stack.discard(action)
            raise ArtifactGapError(needed_artifact, action.value)

        try:
            sid = self._add_step(action, contract, parameters={})
        finally:
            self._injection_stack.discard(action)

        self._decisions.append(PlanningDecision(
            step_id=sid,
            action_type=action,
            decision="added",
            reason=f"自动注入以提供 {needed_artifact} 给下游步骤",
        ))

    def _add_step(self, action: ActionType, contract: ActionIOContract | None, *,
                  parameters: dict[str, Any], gpu_required: bool | None = None) -> str:
        inputs, deps = self._ensure_inputs(action, contract)
        sid = self._unique_step_id(action)
        outputs = [self._artifact_ref(key) for key in contract.outputs] if contract else []
        step = WorkflowStep(
            step_id=sid,
            action_type=action,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            quality_gates=list(contract.default_quality_gates) if contract else [],
            resource_profile=self._resource_profile(action, contract, gpu_required=gpu_required),
        )
        self._steps[sid] = step
        self._step_dependencies[sid] = set(deps)

        if contract:
            for output_key in contract.outputs:
                self._available_artifacts[output_key] = sid
        return sid

    def _ensure_mandatory_steps(self) -> None:
        """确保强制步骤存在。"""
        from autovs.dag import SCREENING_LIBRARY, TARGET_STRUCTURE

        # input_validation
        if ActionType.INPUT_VALIDATION not in {
            self._steps[s].action_type for s in self._steps
        }:
            contract = get_contract(ActionType.INPUT_VALIDATION)
            self._add_step(ActionType.INPUT_VALIDATION, contract, parameters={})
            self._decisions.append(PlanningDecision(
                step_id="input-validation",
                action_type=ActionType.INPUT_VALIDATION,
                decision="added",
                reason="服务拥有的输入校验步骤",
            ))

        # target_structure_acquisition（需要时）
        if TARGET_STRUCTURE not in self._available_artifacts:
            if self._capacity(ActionType.TARGET_STRUCTURE_ACQUISITION) != "unavailable":
                contract = get_contract(ActionType.TARGET_STRUCTURE_ACQUISITION)
                sid = self._add_step(ActionType.TARGET_STRUCTURE_ACQUISITION, contract, parameters={})
                self._decisions.append(PlanningDecision(
                    step_id=sid,
                    action_type=ActionType.TARGET_STRUCTURE_ACQUISITION,
                    decision="added",
                    reason="服务拥有的靶结构获取步骤",
                ))

        # 只有已有打分产物时才补最终排序；纯结构分析任务不强塞筛选收尾。
        from autovs.dag import SCORES_CSV
        if SCORES_CSV in self._available_artifacts and ActionType.FINAL_RANKING not in {
            step.action_type for step in self._steps.values()
        }:
            contract = get_contract(ActionType.FINAL_RANKING)
            sid = self._add_step(ActionType.FINAL_RANKING, contract, parameters={})
            self._decisions.append(PlanningDecision(
                step_id=sid,
                action_type=ActionType.FINAL_RANKING,
                decision="added",
                reason="已有 scores_csv，自动补最终排序",
            ))

    def _topological_sort(self) -> list[str]:
        """稳定拓扑排序。"""
        # 构建依赖图
        in_degree: dict[str, int] = {sid: 0 for sid in self._steps}
        deps: dict[str, set[str]] = {sid: set() for sid in self._steps}

        for sid, frozen_deps in self._step_dependencies.items():
            deps[sid].update(dep for dep in frozen_deps if dep in self._steps and dep != sid)

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
            deps = self._step_dependencies.get(sid, set())
            step.requires = sorted(dep for dep in deps if dep in self._steps and dep != sid)

    def _validate_constraints(self, ordered: list[str]) -> None:
        if len(ordered) > self.constraints.max_steps:
            raise PlanningValidationError(
                f"plan has {len(ordered)} steps, exceeding max_steps={self.constraints.max_steps}"
            )
        actions = {self._steps[sid].action_type for sid in ordered}
        missing_required = [a.value for a in self.constraints.required_actions if a not in actions]
        if missing_required:
            raise PlanningValidationError(f"required actions missing from plan: {missing_required}")
        forbidden_present = [a.value for a in self.constraints.forbidden_actions if a in actions]
        if forbidden_present:
            raise PlanningValidationError(f"forbidden actions present in plan: {forbidden_present}")
        total_cost = 0.0
        total_walltime = 0
        total_cpu_hours = 0.0
        total_gpu_hours = 0.0
        max_risk = 0.0
        for sid in ordered:
            step = self._steps[sid]
            cost = estimate_step_cost(
                step.action_type,
                cpus=step.resource_profile.cpus,
                gpu_required=step.resource_profile.gpu_required,
            )
            risk = estimate_step_risk(step.action_type, self._capacity(step.action_type))
            total_cost += cost.relative_cost
            total_walltime += cost.estimated_walltime_seconds or 0
            total_cpu_hours += cost.estimated_cpu_hours or 0.0
            total_gpu_hours += cost.estimated_gpu_hours or 0.0
            max_risk = max(max_risk, risk.failure_probability)
        if self.constraints.max_relative_cost is not None and total_cost > self.constraints.max_relative_cost:
            raise PlanningValidationError("plan exceeds max_relative_cost")
        if self.constraints.max_walltime_seconds is not None and total_walltime > self.constraints.max_walltime_seconds:
            raise PlanningValidationError("plan exceeds max_walltime_seconds")
        if self.constraints.max_cpu_hours is not None and total_cpu_hours > self.constraints.max_cpu_hours:
            raise PlanningValidationError("plan exceeds max_cpu_hours")
        if self.constraints.max_gpu_hours is not None and total_gpu_hours > self.constraints.max_gpu_hours:
            raise PlanningValidationError("plan exceeds max_gpu_hours")
        if self.constraints.max_failure_risk is not None and max_risk > self.constraints.max_failure_risk:
            raise PlanningValidationError("plan exceeds max_failure_risk")
