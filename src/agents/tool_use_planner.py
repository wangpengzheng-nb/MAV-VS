"""ToolUsePlannerAgent — LLM 建议 + 确定性图构建。

混合架构：
1. LLM（可选）生成 PlannerDraft（仅 action 意图 + 重要性）
2. 确定性 WorkflowGraphBuilder 构建可执行 WorkflowPlan
3. 严格校验 + 规划报告

无 LLM 时从 strategy 中启发式提取 action 意图。
"""

from __future__ import annotations

import json
import os
from typing import Any

from autovs.capabilities import list_capabilities
from autovs.config import Settings
from autovs.planning.contracts import ACTION_CONTRACTS, get_contract
from autovs.planning.errors import PlannerError
from autovs.planning.graph_builder import (
    PlannedActionIntent, PlannerConstraints, PlannerDraft, PlannerResult,
    WorkflowGraphBuilder,
)
from autovs.schemas import ActionType, InputManifest


# ═══════════════════════════════════════════════════════════════════════
# ToolUsePlannerAgent
# ═══════════════════════════════════════════════════════════════════════

class ToolUsePlannerAgent:
    """工具使用规划智能体。

    支持有 LLM 和无 LLM 两种模式。无 LLM 时从 strategy 中启发式提取。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: Any = None,
        model: str = "deepseek-chat",
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.model = model

    def plan(
        self,
        strategy: dict[str, Any],
        input_manifest: InputManifest,
        capabilities: list[Any] | None = None,
        constraints: PlannerConstraints | None = None,
    ) -> PlannerResult:
        """主规划入口。

        Args:
            strategy: 选中的 strategy dict
            input_manifest: 输入清单
            capabilities: 能力列表（None=自动获取）
            constraints: 规划约束（None=默认）

        Returns:
            PlannerResult（含可执行 WorkflowPlan）
        """
        if constraints is None:
            constraints = PlannerConstraints()
        if capabilities is None and self.settings is not None:
            capabilities = list_capabilities(self.settings)
        elif capabilities is None:
            capabilities = []

        # 1. 生成 PlannerDraft（LLM 或启发式）
        draft = self._make_draft(strategy, input_manifest, capabilities)

        # 2. 确定性图构建
        builder = WorkflowGraphBuilder(
            draft=draft,
            input_manifest=input_manifest,
            capabilities=capabilities,
            constraints=constraints,
            settings=self.settings,
        )
        result = builder.build()

        # 3. 追加 LLM 相关警告
        for w in draft.warnings:
            if w not in result.warnings:
                result.warnings.append(w)

        return result

    # ── Draft 生成 ──────────────────────────────────────────────────

    def _make_draft(
        self,
        strategy: dict[str, Any],
        manifest: InputManifest,
        capabilities: list[Any],
    ) -> PlannerDraft:
        """生成 PlannerDraft：优先 LLM，失败时启发式。"""
        strategy_id = str(strategy.get("strategy_id") or strategy.get("strategy_name", "unknown"))

        if self.llm_client is not None:
            try:
                return self._llm_draft(strategy, manifest, capabilities)
            except Exception:
                pass

        return self._heuristic_draft(strategy, manifest)

    def _llm_draft(
        self,
        strategy: dict[str, Any],
        manifest: InputManifest,
        capabilities: list[Any],
    ) -> PlannerDraft:
        """调用 LLM 生成 PlannerDraft。

        LLM 只能输出 action_type、importance、parameters、rationale。
        不允许输出路径、URL、shell 命令。
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise PlannerError("LLM draft requested but no API key available")

        from openai import OpenAI
        from autovs.planning.scoring import (
            RELATIVE_COSTS, FAILURE_RISKS, WALLTIME_ESTIMATES,
        )

        cap_list = []
        for c in capabilities:
            cap_list.append({
                "action_type": c.action_type.value,
                "availability": c.availability,
                "description": c.description,
                "gpu_required": c.gpu_required,
            })

        contract_list = []
        for action, contract in ACTION_CONTRACTS.items():
            contract_list.append({
                "action_type": action.value,
                "scientific_role": contract.scientific_role,
                "required_inputs": contract.required_inputs,
                "outputs": contract.outputs,
            })

        prompt = json.dumps({
            "task": "给定一个虚拟筛选策略，提取规划意图。",
            "rules": [
                "只输出 action_type、importance (required/recommended/optional)、parameters、rationale",
                "不得输出文件路径、URL、shell命令、conda命令",
                "不得编造不存在的 action_type",
                "不得修改用户上传的锁定资产",
                "importance 规则: 核心步骤=required, 增强步骤=recommended, 可选项=optional",
            ],
            "strategy": {
                "name": strategy.get("strategy_name", ""),
                "description": strategy.get("description", "")[:500],
                "pipeline": [
                    {
                        "action_type": str(s.get("action_type", "")),
                        "description": str(s.get("description", ""))[:200],
                    }
                    for s in (strategy.get("pipeline") or strategy.get("updated_pipeline") or [])
                ][:15],
            },
            "available_actions": contract_list,
            "capabilities": cap_list,
        }, ensure_ascii=False)

        client = OpenAI(
            api_key=api_key,
            base_url=f"{os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com').rstrip('/')}/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是虚拟筛选规划助手。只输出 JSON。"
                        "字段: strategy_id, scientific_objectives, warnings, actions。"
                        "每个 action 含: action_type, importance, parameters, rationale。"
                        "禁止输出路径、URL、shell命令。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = json.loads(response.choices[0].message.content or "{}")
        return self._parse_draft(raw, strategy, manifest)

    def _heuristic_draft(
        self,
        strategy: dict[str, Any],
        manifest: InputManifest,
    ) -> PlannerDraft:
        """从 strategy 中启发式提取 PlannerDraft。"""
        strategy_id = str(strategy.get("strategy_id") or strategy.get("strategy_name", "unknown"))
        pipeline = (
            strategy.get("updated_pipeline")
            or strategy.get("pipeline")
            or strategy.get("pipeline_steps")
            or []
        )

        actions: list[PlannedActionIntent] = []
        warnings: list[str] = []

        for raw_step in pipeline:
            if not isinstance(raw_step, dict):
                continue
            raw_action = str(raw_step.get("action_type", "")).strip()

            # 通过 compiler ALIASES 映射
            from autovs.compiler import ALIASES
            if raw_action in ALIASES:
                action = ALIASES[raw_action]
            else:
                try:
                    action = ActionType(raw_action)
                except ValueError:
                    warnings.append(f"未知 action_type: {raw_action}")
                    continue

            # 跳过服务拥有的步骤（编译时会被注入）
            contract = get_contract(action)
            if contract and contract.service_owned:
                continue

            # 判断重要性
            desc = str(raw_step.get("description", "")).lower()
            params = raw_step.get("parameters") or raw_step.get("params") or {}
            if isinstance(params, dict):
                params = {k: v for k, v in params.items()
                          if isinstance(v, (str, int, float, bool, list))}

            if any(kw in desc for kw in ("必须", "核心", "required", "essential")):
                importance = "required"
            elif any(kw in desc for kw in ("可选", "optional", "增强", "further")):
                importance = "optional"
            else:
                importance = "recommended"

            actions.append(PlannedActionIntent(
                action_type=action,
                importance=importance,
                parameters=params,
                rationale=str(raw_step.get("description", ""))[:300],
            ))

        # 强制要求关键步骤
        has_docking = any(a.action_type == ActionType.MOLECULAR_DOCKING for a in actions)
        if has_docking:
            for required_action in [
                ActionType.POCKET_DEFINITION,
                ActionType.PROTEIN_PREPARATION,
            ]:
                if not any(a.action_type == required_action for a in actions):
                    actions.append(PlannedActionIntent(
                        action_type=required_action,
                        importance="required",
                        rationale="对接必须的步骤（自动补充）",
                    ))

        return PlannerDraft(
            strategy_id=strategy_id,
            actions=actions,
            scientific_objectives=[strategy.get("description", "")[:500]],
            warnings=warnings,
        )

    def _parse_draft(
        self,
        raw: dict[str, Any],
        strategy: dict[str, Any],
        manifest: InputManifest,
    ) -> PlannerDraft:
        """解析并校验 LLM 输出的 draft JSON。"""
        strategy_id = str(strategy.get("strategy_id") or strategy.get("strategy_name", "unknown"))
        warnings: list[str] = []
        actions: list[PlannedActionIntent] = []

        raw_actions = raw.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        for item in raw_actions:
            if not isinstance(item, dict):
                warnings.append(f"跳过非字典 action: {item}")
                continue

            action_str = str(item.get("action_type", "")).strip()
            try:
                action = ActionType(action_str)
            except ValueError:
                warnings.append(f"LLM 返回未知 action_type: {action_str}，跳过")
                continue

            importance = str(item.get("importance", "recommended")).lower()
            if importance not in {"required", "recommended", "optional"}:
                importance = "recommended"

            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            # 安全过滤：拒绝路径、URL、shell
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    if v.startswith(("/", "~", "C:\\", "http://", "https://")):
                        warnings.append(f"LLM 参数 {k}={v} 疑似路径/URL，已移除")
                        continue
                    if any(cmd in v.lower() for cmd in ("rm ", "bash", "sh -c", ";", "&&")):
                        warnings.append(f"LLM 参数 {k} 疑似 shell 命令，已移除")
                        continue
                safe_params[k] = v

            rationale = str(item.get("rationale", ""))[:500]

            actions.append(PlannedActionIntent(
                action_type=action,
                importance=importance,
                parameters=safe_params,
                rationale=rationale,
            ))

        return PlannerDraft(
            strategy_id=strategy_id,
            actions=actions,
            scientific_objectives=raw.get("scientific_objectives", []),
            warnings=warnings + raw.get("warnings", []),
        )
