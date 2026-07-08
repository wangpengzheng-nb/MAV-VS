"""
AutoVS-Agent v2.0: Expert Committee — 红军辩论评审团 (Red Team)
================================================================
职责: 作为"杠精"评审团，从三个维度攻击候选虚拟筛选策略。
     三人设同时出击: 药化老兵 / 漏斗终结者 / 靶点特异性专家。

输入: 两个候选策略 + 靶点画像
输出: ExpertAttack × 6 (每个策略被3个专家攻击)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. Pydantic Schema
# =============================================================================

class ExpertAttackSchema(BaseModel):
    persona: str = Field(..., description="medchem_veteran / funnel_terminator / target_specialist")
    persona_name: str = Field(..., description="可读名称: 药化老兵 / 漏斗终结者 / 靶点专家")
    attack_points: List[str] = Field(
        ..., min_length=1, description="具体的攻击点，每条 20-80 字"
    )
    severity: str = Field(..., description="critical / major / minor")
    suggested_fixes: List[str] = Field(default_factory=list, description="改进建议")
    agreement_with_strategy: float = Field(
        ..., ge=0.0, le=1.0, description="对该策略的整体认可度 (0=完全不同意, 1=完全同意)"
    )


class ExpertDebateOutput(BaseModel):
    """红军对一对策略的完整评审。"""
    debate_summary: str = Field(..., description="本轮辩论的总体摘要 (150-300字)")
    attacks_on_strategy_a: List[ExpertAttackSchema] = Field(
        ..., min_length=3, max_length=3, description="三位专家对策略A的攻击"
    )
    attacks_on_strategy_b: List[ExpertAttackSchema] = Field(
        ..., min_length=3, max_length=3, description="三位专家对策略B的攻击"
    )


ExpertDebateOutput.model_rebuild()
ExpertAttackSchema.model_rebuild()


# =============================================================================
# 2. 红军系统提示词
# =============================================================================

RED_TEAM_SYSTEM_PROMPT = """\
# 人设: 红军虚拟筛选评审团 (Red Team Review Panel)

你是一个由三位顶级专家组成的**红军评审团**。
你的任务是同时扮演以下三个人设，对两个候选虚拟筛选策略进行**无情的批判性评审**。

---

## 人设 1: 药化老兵 (MedChem Veteran)
**身份**: 在各大药企工作了 25 年的药物化学家，亲眼见证了无数失败的虚拟筛选项目。
**审查角度**:
- ADMET 和类药性: 策略是否过于宽松导致筛选出的分子根本不可成药?
- PAINS 和假阳性: 策略是否有效排除了已知的假阳性干扰物?
- 合成可行性和成本
- 是否过于依赖单一的对接分数? (对接分数和实验活性相关性很差!)
**攻击风格**: 经验主义、尖刻但务实。"这个策略在 2008 年我们就试过了，根本行不通。"

## 人设 2: 漏斗终结者 (Funnel Terminator)
**身份**: 计算化学家，专门分析虚拟筛选漏斗的效率和存活率。
**审查角度**:
- 存活率审视: "按这个策略的绝对过滤条件, 10 万分子能活下来多少? 如果 < 10 个, 这个策略就是废的!"
- 每个过滤条件的淘汰率估算
- 策略的应急预案是否充分、步骤是否合理
- 宽松策略会产生多少假阳性, 下游负担是否可承受
**攻击风格**: 数据驱动、冷血无情。"这个条件的组合会让存活率 < 0.01%，你这是在浪费所有人的时间。"

## 人设 3: 靶点特异性专家 (Target Specialist)
**身份**: 专门研究该靶点类型的结构生物学家/药理学家。
**审查角度**:
- 靶点适配性: 策略的方法学是否适合这个靶点?
  * PPI 靶点用传统 Ro5 → 完全不对!
  * 隐蔽口袋用刚性对接 → 不会有效果!
- 关键相互作用的覆盖: 策略是否考虑了已知的药效团要素?
- 选择性: 策略是否能区分靶点亚型?
- 技术路线的合理性
**攻击风格**: 学术严谨、引用文献。"根据 XXX 论文, 该靶点的口袋在 MD 中会显著构象变化, 刚性对接毫无意义。"

---

# 评审规则
1. 每位专家必须对该策略给出一个 agreement_with_strategy 分数 (0.0-1.0)
2. attack_points 必须具体、有建设性，不能是泛泛的 "这个策略不好"
3. 如果某个策略确实存在致命缺陷 (如存活率接近 0)，请在 severity 中标为 "critical"
4. 针对靶点类型错误的策略 (如 PPI 用 Ro5) 必须严厉批评

---

# 输出格式
严格输出 ExpertDebateOutput JSON Schema，包含 6 个 ExpertAttack (每个策略 3 个)。
"""


# =============================================================================
# 3. RedTeamReviewer
# =============================================================================

class RedTeamReviewer:
    """红军评审团 — 三位专家同时攻击两个候选策略。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 4096,
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

    def debate_strategies(
        self,
        strategy_a: dict,
        strategy_b: dict,
        target_profile: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """对两个策略进行红军评审。

        Returns:
            {
                "attacks_on_a": List[ExpertAttack],
                "attacks_on_b": List[ExpertAttack],
                "debate_summary": str,
            }
        """
        user_prompt = self._build_debate_prompt(strategy_a, strategy_b, target_profile)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": RED_TEAM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"请严格输出 JSON:\n{json.dumps(ExpertDebateOutput.model_json_schema(), indent=2, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            validated = ExpertDebateOutput.model_validate(parsed)

            return {
                "attacks_on_a": [a.model_dump() for a in validated.attacks_on_strategy_a],
                "attacks_on_b": [a.model_dump() for a in validated.attacks_on_strategy_b],
                "debate_summary": validated.debate_summary,
            }
        except Exception as e:
            return self._fallback_debate(strategy_a, strategy_b, str(e))

    def _build_debate_prompt(self, sa: dict, sb: dict, profile: Optional[dict] = None) -> str:
        def fmt_strategy(s: dict) -> str:
            af = s.get("absolute_filters", [])
            rr = s.get("relative_rankings", [])
            cp = s.get("contingency_plan", {})
            return f"""\
**{s.get('strategy_name', '?')}** — {s.get('strategy_tagline', '')}
- 方法: {s.get('approach_type', '?')}
- 原理: {s.get('rationale', '')[:300]}
- 绝对过滤 ({len(af)}条): {', '.join(f.get('description','')[:40] for f in af)}
- 相对排序 ({len(rr)}条): {', '.join(f.get('description','')[:40] for f in rr)}
- 应急预案触发: {cp.get('trigger_condition', '?')}
- 估算存活率: {s.get('estimated_survival_rate', '?')}
- 优势: {s.get('strengths', [])}
- 劣势: {s.get('weaknesses', [])}"""

        target_ctx = ""
        if profile:
            sa_info = profile.get("structural_assessment", {})
            target_ctx = f"""
### 靶点上下文
- 口袋类型: {sa_info.get('pocket_type', '?')}
- 口袋体积: {sa_info.get('pocket_volume_estimate', '?')}
- 有共晶结构: {sa_info.get('has_cocrystal_with_ligand', '?')}
- 主要指标: {profile.get('priority_metrics', {}).get('primary_metrics', [])}
"""

        return f"""\
## 红军辩论任务

{target_ctx}

### 策略 A
{fmt_strategy(sa)}

### 策略 B
{fmt_strategy(sb)}

### 评审要求
请三位专家 (药化老兵、漏斗终结者、靶点专家) 分别对两个策略进行批判性评审。
"""

    def _fallback_debate(self, sa: dict, sb: dict, error: str) -> Dict[str, Any]:
        """LLM 不可用时的降级评审。"""
        def basic_attack(persona: str, pname: str, strategy_name: str) -> dict:
            return {
                "persona": persona,
                "persona_name": pname,
                "attack_points": [f"LLM不可用，无法对{strategy_name}生成详细攻击 ({error[:80]})"],
                "severity": "minor",
                "suggested_fixes": ["请人工审查该策略"],
                "agreement_with_strategy": 0.5,
            }

        return {
            "attacks_on_a": [
                basic_attack("medchem_veteran", "药化老兵", sa.get("strategy_name", "A")),
                basic_attack("funnel_terminator", "漏斗终结者", sa.get("strategy_name", "A")),
                basic_attack("target_specialist", "靶点专家", sa.get("strategy_name", "A")),
            ],
            "attacks_on_b": [
                basic_attack("medchem_veteran", "药化老兵", sb.get("strategy_name", "B")),
                basic_attack("funnel_terminator", "漏斗终结者", sb.get("strategy_name", "B")),
                basic_attack("target_specialist", "靶点专家", sb.get("strategy_name", "B")),
            ],
            "debate_summary": f"[FALLBACK] LLM 调用失败: {error[:200]}",
        }


# =============================================================================
# LangGraph 节点
# =============================================================================

def red_team_debate_node(state: dict) -> dict:
    """锦标赛辩论节点: 取配对队列下一个 → 红军评审 → 记录。"""
    from datetime import datetime as dt

    ts = state.get("tournament_state", {})
    pairings = list(ts.get("pairings_queue", []))

    if not pairings:
        return {"pipeline_stage": "tournament", "event_log": ["[RedTeam] No more pairings — tournament complete."]}

    # 取出下一对
    pair = pairings.pop(0)
    name_a, name_b = pair[0], pair[1]

    # 查找策略详情
    strategies = {s["strategy_name"]: s for s in state.get("candidate_strategies", [])}
    sa = strategies.get(name_a, {})
    sb = strategies.get(name_b, {})

    if not sa or not sb:
        return {"tournament_state": {**ts, "pairings_queue": pairings}}

    reviewer = RedTeamReviewer()
    result = reviewer.debate_strategies(
        strategy_a=sa, strategy_b=sb,
        target_profile=state.get("target_profile"),
    )

    now = dt.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "tournament",
        "tournament_state": {
            **ts,
            "pairings_queue": pairings,
        },
        "event_log": [
            f"[{now}] [RedTeam] Debated: {name_a} vs {name_b}. "
            f"Summary: {result.get('debate_summary', '')[:120]}"
        ],
        # 临时存储当前辩论结果 (供 judge_node 读取)
        "_current_debate_pair": [name_a, name_b],
        "_current_debate_result": result,
        "updated_at": now,
    }
