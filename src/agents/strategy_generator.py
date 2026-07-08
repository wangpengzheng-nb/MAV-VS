"""
AutoVS-Agent v2.0: Strategy Generator (策略生成器)
===================================================
职责: 基于 TargetProfile 生成 3-5 个差异化的虚拟筛选策略。
     每个策略必须包含绝对过滤条件、相对排序条件、软指标和应急预案。

核心约束:
  - 严禁使用绝对分数作为唯一标准!
  - 必须区分 absolute_filters / relative_rankings / soft_metrics 三类规则
  - 每个策略必须有 contingency_plan
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. 结构化输出 Schema
# =============================================================================

class FilterRuleSchema(BaseModel):
    rule_id: str = Field(..., description="规则唯一标识: filter_01, rank_01 等")
    category: str = Field(
        ..., description="absolute_filter / relative_ranking / soft_metric"
    )
    description: str = Field(..., description="人类可读的规则描述")
    parameter: str = Field(..., description="参数名: MW, LogP, docking_score, PAINS 等")
    operator: str = Field(..., description="运算符: <, >, <=, >=, in_range, top_percentile, must_not_match, should_match")
    value: Any = Field(..., description="阈值或取值范围")
    rationale: str = Field(..., description="科学依据")
    relaxable: bool = Field(default=True, description="是否可在应急时放宽")
    relaxed_value: Optional[Any] = Field(default=None, description="放宽后的值")


class ContingencyPlanSchema(BaseModel):
    trigger_condition: str = Field(
        ..., description="触发应急预案的条件，如 'survivors < 10'"
    )
    relaxation_steps: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="有序的放宽步骤列表。每步: {rule_id, new_value, reason}"
    )
    minimum_acceptable_thresholds: Dict[str, Any] = Field(
        ..., description="底线阈值，超过此值则策略不可用"
    )
    fallback_strategy: str = Field(
        ..., description="如果所有放宽步骤都失败, 应该做什么"
    )


class CandidateStrategySchema(BaseModel):
    strategy_name: str = Field(
        ..., description="策略名称，如 '保守SBDD策略' / '激进bRo5策略' / 'ML优先策略'"
    )
    strategy_tagline: str = Field(
        ..., description="一句话总结，如 '高门槛，低假阳性，适合有共晶结构的经典靶点'"
    )
    rationale: str = Field(
        ..., description="为什么这个策略适合该靶点 (200-400字)"
    )
    approach_type: str = Field(
        ..., description="structure_based / ligand_based / hybrid / ml_driven"
    )
    absolute_filters: List[FilterRuleSchema] = Field(
        ...,
        min_length=1,
        description="绝对过滤条件 (一票否决)。如 PAINS, 毒性基团, MW>1000 等"
    )
    relative_rankings: List[FilterRuleSchema] = Field(
        ...,
        min_length=1,
        description="相对排序条件 (择优而非淘汰)。如 对接分数前5%, LogP最优区间等"
    )
    soft_metrics: List[FilterRuleSchema] = Field(
        default_factory=list,
        description="加分/减分软指标。如 QED>0.5, TPSA<140 加5分 等"
    )
    contingency_plan: ContingencyPlanSchema = Field(
        ..., description="应急预案"
    )
    estimated_survival_rate: str = Field(
        ..., description="估算的存活率，如 '~5% → ~5000 survivors from 100K'"
    )
    strengths: List[str] = Field(..., min_length=1, description="本策略的优势")
    weaknesses: List[str] = Field(..., min_length=1, description="本策略的劣势")


class StrategyGenerationOutput(BaseModel):
    """策略生成器的完整输出: 3-5 个差异化策略。"""

    generation_rationale: str = Field(
        ..., description="为什么生成这几个差异化的策略 (200-400字)"
    )
    strategies: List[CandidateStrategySchema] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 个差异化虚拟筛选策略",
    )


StrategyGenerationOutput.model_rebuild()
CandidateStrategySchema.model_rebuild()
ContingencyPlanSchema.model_rebuild()
FilterRuleSchema.model_rebuild()


# =============================================================================
# 2. 核心系统提示词
# =============================================================================

STRATEGY_GENERATOR_SYSTEM_PROMPT = """\
# 人设: 虚拟筛选策略架构师 (VS Strategy Architect)

你是一位在制药工业界有 15 年经验的**虚拟筛选策略架构师**。
你的任务是基于靶点情报分析师提供的 TargetProfile，
设计 **3-5 个差异化 (Divergent) 的虚拟筛选策略**。

---

# 策略设计原则 (Critical!)

## 1. 差异化要求
你生成的策略必须覆盖**不同的方法论角度**:
- **保守策略**: 严格过滤，低假阳性率，适合有高质量结构的靶点
- **探索策略**: 放宽阈值，宁愿多筛不漏，适合缺乏已知配体的困难靶点
- **混合策略**: 结合对接和机器学习预测
- **配体中心策略**: 基于已知配体的药效团/相似性搜索
- **片段中心策略**: 低分子量片段库优先，适合 FBDD

## 2. 三类规则严格区分
绝对不要混为一谈！每个策略必须清晰区分:

### A. 绝对过滤条件 (absolute_filters)
- 一票否决! 满足任一条件立即淘汰
- 例子: PAINS 子结构、已知毒性基团、MW>1000、不可合成的分子
- 这些条件即使导致存活率为 0 也不能放松 — 它们是科学红线

### B. 相对排序条件 (relative_rankings)
- 不直接淘汰! 用于排序和选取 Top-N
- 例子: 对接分数前 5%、LogP 在 2-4 范围内的优先
- 核心原则: **严禁使用绝对分数作为唯一标准!**
  不要写 "docking_score < -9.5" (这是绝对过滤!)
  应该写 "docking_score 排名前 10%" (相对!)
- 如果必须使用数值阈值，必须在 rationale 中充分说明科学依据

### C. 软指标 (soft_metrics)
- 加分/减分项，综合评分时参考
- 例子: QED > 0.5 (+5分), TPSA > 140 (-3分)

## 3. 应急预案 (Contingency Plan) — 强制要求!
每个策略必须包含详细的应急预案:
- **触发条件**: 明确写 "survivors < 10" 或 "survivors < 1%"
- **放宽步骤**: 有序列表，先放宽相对排序条件，最后才动绝对过滤条件
- **底线阈值**: 写明确认哪些条件绝对不能放宽 (如 PAINS 红线)
- **兜底方案**: 如果放宽后仍然失败，该怎么办?

---

# 输出格式
严格输出 StrategyGenerationOutput JSON Schema。
"""


# =============================================================================
# 3. StrategyGeneratorAgent
# =============================================================================

class StrategyGeneratorAgent:
    """策略生成器 — 基于靶点画像生成多个差异化虚拟筛选策略。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 8192,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.api_base}/v1")
        return self._client

    def generate_strategies(
        self,
        target_profile: dict,
        target_info: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """生成 3-5 个差异化策略。

        Args:
            target_profile: TargetProfile dict (来自 Target Scout)。
            target_info: 原始 TargetInfo (可选)。

        Returns:
            {
                "strategies": List[CandidateStrategy],
                "generation_rationale": str,
            }
        """
        user_prompt = self._build_prompt(target_profile, target_info)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": STRATEGY_GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"请严格输出 JSON:\n{json.dumps(StrategyGenerationOutput.model_json_schema(), indent=2, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            validated = StrategyGenerationOutput.model_validate(parsed)

            return {
                "strategies": [s.model_dump() for s in validated.strategies],
                "generation_rationale": validated.generation_rationale,
            }
        except Exception as e:
            # 降级: 返回 3 个基本策略
            return self._fallback_strategies(target_profile, str(e))

    def _build_prompt(self, profile: dict, target_info: Optional[dict] = None) -> str:
        """构建策略生成 User Prompt。"""
        sa = profile.get("structural_assessment", {})
        kl = profile.get("known_ligand_info", {})
        pm = profile.get("priority_metrics", {})

        return f"""\
## 策略生成任务

### 靶点画像摘要
- **靶点**: {profile.get('target_name', 'Unknown')}
- **口袋类型**: {sa.get('pocket_type', 'unknown')}
- **口袋体积**: {sa.get('pocket_volume_estimate', 'unknown')}
- **口袋极性**: {sa.get('pocket_polarity', 'unknown')}
- **有共晶结构**: {sa.get('has_cocrystal_with_ligand', False)}
- **有已知配体**: {kl.get('has_known_active_ligands', False)}
- **已知配体亲和力**: {kl.get('binding_affinity_range', 'unknown')}
- **蛋白柔性**: {sa.get('flexibility_concern', 'unknown')}

### 优先级指标
- **主要指标**: {pm.get('primary_metrics', [])}
- **红线**: {pm.get('red_flags', [])}
- **建议阈值**: {pm.get('suggested_thresholds', {})}

### 设计挑战
{profile.get('drug_design_challenges', [])}

### 推荐技术路线
{profile.get('recommended_approaches', [])}

### 任务
请基于以上靶点画像，设计 3-5 个差异化的虚拟筛选策略。
每个策略必须包含完整的过滤规则和应急预案。
"""

    def _fallback_strategies(self, profile: dict, error: str) -> Dict[str, Any]:
        """LLM 不可用时的降级策略集。"""
        sa = profile.get("structural_assessment", {})
        pocket_type = sa.get("pocket_type", "unknown")

        strategies = [
            {
                "strategy_name": "保守SBDD策略",
                "strategy_tagline": "严格过滤，低假阳性",
                "rationale": "基于结构的保守策略。利用对接和理化性质严格筛选。",
                "approach_type": "structure_based",
                "absolute_filters": [
                    {"rule_id": "abs_01", "category": "absolute_filter", "description": "排除PAINS子结构", "parameter": "PAINS", "operator": "must_not_match", "value": "any", "rationale": "假阳性干扰", "relaxable": False, "relaxed_value": None},
                    {"rule_id": "abs_02", "category": "absolute_filter", "description": "排除已知毒性基团", "parameter": "toxicity_SMARTS", "operator": "must_not_match", "value": "any", "rationale": "安全性红线", "relaxable": False, "relaxed_value": None},
                ],
                "relative_rankings": [
                    {"rule_id": "rel_01", "category": "relative_ranking", "description": "对接分数前15%", "parameter": "docking_score", "operator": "top_percentile", "value": 15, "rationale": "择优而非淘汰", "relaxable": True, "relaxed_value": 25},
                ],
                "soft_metrics": [
                    {"rule_id": "soft_01", "category": "soft_metric", "description": "QED>0.5 加分", "parameter": "QED", "operator": ">", "value": 0.5, "rationale": "类药性偏好", "relaxable": True, "relaxed_value": 0.3},
                ],
                "contingency_plan": {
                    "trigger_condition": "survivors < 10",
                    "relaxation_steps": [{"rule_id": "rel_01", "new_value": 25, "reason": "放宽排序阈值"}, {"rule_id": "soft_01", "new_value": 0.3, "reason": "放宽软指标"}],
                    "minimum_acceptable_thresholds": {"MW": 1000, "LogP": 8},
                    "fallback_strategy": "切换到探索策略",
                },
                "estimated_survival_rate": "~3% (~3000 from 100K)",
                "strengths": ["低假阳性率", "SBDD黄金标准"],
                "weaknesses": ["可能漏掉非经典结合模式", f"口袋类型={pocket_type}可能不适用"],
            },
            {
                "strategy_name": "探索性宽松策略",
                "strategy_tagline": "宽松阈值，多筛不漏",
                "rationale": "对于口袋特征不明确的靶点，宽松过滤避免假阴性。",
                "approach_type": "hybrid",
                "absolute_filters": [
                    {"rule_id": "abs_01", "category": "absolute_filter", "description": "排除PAINS子结构", "parameter": "PAINS", "operator": "must_not_match", "value": "any", "rationale": "假阳性干扰", "relaxable": False, "relaxed_value": None},
                ],
                "relative_rankings": [
                    {"rule_id": "rel_01", "category": "relative_ranking", "description": "对接分数前30%", "parameter": "docking_score", "operator": "top_percentile", "value": 30, "rationale": "宽筛", "relaxable": True, "relaxed_value": 50},
                ],
                "soft_metrics": [],
                "contingency_plan": {
                    "trigger_condition": "survivors < 10",
                    "relaxation_steps": [{"rule_id": "rel_01", "new_value": 50, "reason": "扩大筛选范围"}],
                    "minimum_acceptable_thresholds": {},
                    "fallback_strategy": "只保留PAINS过滤，其他全部通过",
                },
                "estimated_survival_rate": "~15% (~15000 from 100K)",
                "strengths": ["高召回率", "适合困难靶点"],
                "weaknesses": ["高假阳性率", "下游负担重"],
            },
            {
                "strategy_name": "配体驱动相似性策略",
                "strategy_tagline": "以已知配体为锚点",
                "rationale": "如果有已知活性配体，优先筛选结构相似的分子。",
                "approach_type": "ligand_based",
                "absolute_filters": [
                    {"rule_id": "abs_01", "category": "absolute_filter", "description": "排除PAINS", "parameter": "PAINS", "operator": "must_not_match", "value": "any", "rationale": "假阳性", "relaxable": False, "relaxed_value": None},
                ],
                "relative_rankings": [
                    {"rule_id": "rel_01", "category": "relative_ranking", "description": "与已知配体Tanimoto>0.4优先", "parameter": "tanimoto_similarity", "operator": ">", "value": 0.4, "rationale": "化学空间聚焦", "relaxable": True, "relaxed_value": 0.3},
                    {"rule_id": "rel_02", "category": "relative_ranking", "description": "药效团匹配前20%", "parameter": "pharmacophore_score", "operator": "top_percentile", "value": 20, "rationale": "药效团优先", "relaxable": True, "relaxed_value": 40},
                ],
                "soft_metrics": [],
                "contingency_plan": {
                    "trigger_condition": "survivors < 10",
                    "relaxation_steps": [{"rule_id": "rel_01", "new_value": 0.3, "reason": "降低相似度阈值"}],
                    "minimum_acceptable_thresholds": {"tanimoto_similarity": 0.25},
                    "fallback_strategy": "放弃配体约束，仅依赖对接",
                },
                "estimated_survival_rate": "~5% (~5000 from 100K)",
                "strengths": ["化学空间聚焦", "高富集率"],
                "weaknesses": ["依赖已知配体质量", "可能错过新骨架"],
            },
        ]

        return {
            "strategies": strategies,
            "generation_rationale": f"[FALLBACK] LLM call failed: {error[:200]}. Using 3 default strategies.",
        }


# =============================================================================
# 4. LangGraph 节点
# =============================================================================

def strategy_generation_node(state: dict) -> dict:
    """Step 2: Strategy Generation — 基于靶点画像生成多策略。"""
    from datetime import datetime as dt

    agent = StrategyGeneratorAgent()
    result = agent.generate_strategies(
        target_profile=state.get("target_profile", {}),
        target_info=state.get("target_info"),
    )

    strategies = result.get("strategies", [])

    # 初始化 Elo 积分
    elo_ratings = {}
    for s in strategies:
        elo_ratings[s["strategy_name"]] = state["tournament_state"]["elo_initial_rating"]

    # 构建配对队列 (所有两两组合)
    pairings = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            pairings.append([strategies[i]["strategy_name"], strategies[j]["strategy_name"]])

    now = dt.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "strategy_generation",
        "candidate_strategies": strategies,
        "tournament_state": {
            **state["tournament_state"],
            "round_number": 0,
            "elo_ratings": elo_ratings,
            "pairings_queue": pairings,
            "completed_debates": 0,
            "current_leader": strategies[0]["strategy_name"] if strategies else "",
        },
        "updated_at": now,
        "event_log": [
            f"[{now}] [StrategyGen] {len(strategies)} strategies generated. "
            f"Pairings: {len(pairings)} debates scheduled. "
            f"Rationale: {result.get('generation_rationale', '')[:100]}"
        ],
    }
