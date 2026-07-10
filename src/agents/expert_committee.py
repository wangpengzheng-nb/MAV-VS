"""
AutoVS-Agent v2.0: Red Team Review Panel (红军三人设评审团)
============================================================
三位专家各持深度调研报告, 从不同领域攻击候选策略:
  1. 药化老兵: ADMET/类药性/假阳性/合成可行性
  2. 漏斗终结者: 存活率/过滤效率/统计合理性
  3. 靶点特异性专家: 策略是否对症 (基于调研报告的靶点特征)
"""

from __future__ import annotations

import json, os, re
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. Schema
# =============================================================================

class DomainAttack(BaseModel):
    persona: str = Field(default="", description="medchem_veteran/funnel_terminator/target_specialist")
    persona_name: str = Field(default="", description="可读名称")
    focus_area: str = Field(default="", description="该专家的审查领域")
    attack_points: List[str] = Field(default_factory=list, description="攻击点列表")
    severity: str = Field(default="minor", description="critical/major/minor")
    agreement: float = Field(default=0.5, ge=0.0, le=1.0, description="0=完全不可行, 1=完全认可")
    suggested_fixes: List[str] = Field(default_factory=list)
    reference_to_report: str = Field(default="", description="引用调研报告中支持该攻击的具体发现")


class DebateOutput(BaseModel):
    debate_summary: str = Field(default="", description="本轮辩论总体摘要")
    attacks_on_a: List[DomainAttack] = Field(default_factory=list, description="对策略A的攻击")
    attacks_on_b: List[DomainAttack] = Field(default_factory=list, description="对策略B的攻击")

DebateOutput.model_rebuild()
DomainAttack.model_rebuild()


# =============================================================================
# 2. 三人设系统提示词 (每人关注不同领域)
# =============================================================================

RED_TEAM_PROMPT = """\
# 红军评审团 — 三人设同时评审

你扮演三位专家, 同时审阅两个候选虚拟筛选策略。
每位专家都**已经阅读了靶点深度调研报告**, 评审时必须引用报告中的发现。

---

## 人设 1: 药化老兵 (MedChem Veteran) — 关注 ADMET/类药性/假阳性
**审查清单**:
- 该策略的 ADMET 过滤是否充分? 是否会筛出不可成药的分子?
- PAINS/BRENK 等假阳性子结构是否被有效排除?
- MW/LogP/TPSA 等阈值的设定是否合理? (参考调研报告中的已知配体数值)
- 合成可行性: 筛出的分子能否合成? 复杂度是否合理?
- 该策略是否会漏掉已知的好分子? (参考调研报告中的已知配体)
**引用调研报告**: 报告中的已知配体活性值和理化性质, 报告中的药效团特征

## 人设 2: 漏斗终结者 (Funnel Terminator) — 关注存活率/过滤效率
**审查清单**:
- 每一步过滤后大概能存活多少分子? 整体存活率估算是否合理?
- 如果存活<10, 应急预案是否足够? 放宽步骤是否合理?
- 宽松策略是否会产生过多假阳性导致下游不堪重负?
- 多个过滤条件叠加后是否会产生意外的0存活?
- 每步的计算成本是否可接受? 总时间和资源估算?
**引用调研报告**: 报告中的库大小估算, 报告中的口袋特征对过滤的影响

## 人设 3: 靶点特异性专家 (Target Specialist) — 关注策略是否对症
**审查清单**:
- 该策略的方法是否适合这个靶点类型?
  * PPI靶点用传统Ro5→完全不对!
  * 隐蔽口袋用刚性对接→没有意义!
  * 激酶用配体相似性而不考虑铰链区→方向跑偏!
- 对接盒子中心和尺寸是否合理? (参考调研报告中的建议坐标)
- PLIP分析关注的残基是否和调研报告中的关键残基一致?
- 阳性对照的选择是否合适?
- 该策略的评分函数是否适合该靶点的结合特征?
**引用调研报告**: 报告中的口袋类型/关键残基/建议对接参数/已知配体结合模式

---

# 评审规则
1. 每位专家必须给出 agreement (0.0-1.0) — 对该策略的综合认可度
2. attack_points 至少4条, 每条50-150字, 必须具体引用调研报告数据
3. reference_to_report 必须填写, 说明引用了报告中的哪个具体发现
4. 如果策略与靶点类型明显不匹配, severity 必须为 "critical"
5. suggested_fixes 至少2条, 给出具体可操作的改进方案
"""


# =============================================================================
# 3. RedTeamReviewer
# =============================================================================

class RedTeamReviewer:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.5, max_tokens=16384):
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

    def debate_strategies(self, strategy_a: dict, strategy_b: dict, research_report: Optional[dict] = None) -> Dict[str, Any]:
        # 提取调研报告关键部分
        report_text = ""
        if research_report:
            report_text = (
                f"靶点: {research_report.get('target_name','?')} "
                f"(基因: {research_report.get('gene_symbol','?')}, "
                f"类型: {research_report.get('target_macromolecule_type','?')}, "
                f"物种: {research_report.get('target_organism','?')})\n\n"
                f"### 调研报告全文\n{research_report.get('full_report_text','')[:3000]}"
            )

        def fmt_strategy(s: dict, label: str) -> str:
            steps = s.get("pipeline_steps", [])
            steps_text = ""
            for st in steps:
                steps_text += (f"    Step{st.get('step_number','?')}: {st.get('step_name','?')} "
                              f"[{st.get('tool','?')}] {st.get('action','?')[:100]} "
                              f"指标={st.get('metric','?')} 阈值={st.get('threshold','?')}\n")
            return f"""\
### {label}: {s.get('strategy_name','?')}
标签: {s.get('strategy_tagline','?')}
方法: {s.get('approach_type','?')}
原理: {s.get('rationale','?')[:300]}
步骤:
{steps_text}
存活估算: {s.get('survival_estimate','?')}
应急预案: {s.get('contingency','?')[:200]}
优势: {s.get('strengths',[])}
劣势: {s.get('weaknesses',[])}"""

        prompt = f"""\
## 调研报告摘要
{report_text[:3000]}

{fmt_strategy(strategy_a, '策略A')}
{fmt_strategy(strategy_b, '策略B')}

三位专家分别评审。引用调研报告具体发现。输出完整JSON (所有字段必填)。"""

        try:
            is_reasoner = "reasoner" in self.model.lower()
            kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                          messages=[{"role":"system","content":RED_TEAM_PROMPT},
                                    {"role":"user","content":prompt},
                                    {"role":"system","content":"输出JSON。每专家attack_points至少4条。格式:{\"debate_summary\":\"...\",\"attacks_on_a\":[{\"persona\":\"...\",\"persona_name\":\"...\",\"focus_area\":\"...\",\"attack_points\":[\"p1\",\"p2\"],\"severity\":\"major\",\"agreement\":0.6,\"suggested_fixes\":[\"f1\"],\"reference_to_report\":\"...\"}],\"attacks_on_b\":[...]}"}])
            if not is_reasoner: kwargs.update(temperature=self.temperature, response_format={"type":"json_object"})
            resp = self.client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raw = getattr(resp.choices[0].message, "reasoning_content", "") or ""
                if "{" in raw: raw = raw[raw.find("{"):]
            parsed = self._robust_json_parse(raw.strip())
            d = DebateOutput.model_validate(parsed)
            return {"attacks_on_a": [a.model_dump() for a in d.attacks_on_a],
                    "attacks_on_b": [a.model_dump() for a in d.attacks_on_b],
                    "debate_summary": d.debate_summary}
        except Exception as e:
            return self._fallback(strategy_a, strategy_b, str(e))

    @staticmethod
    def _robust_json_parse(raw: str) -> Dict[str, Any]:
        """多层JSON修复。"""
        try: return json.loads(raw)
        except json.JSONDecodeError: pass
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        try: return json.loads(cleaned)
        except json.JSONDecodeError: pass
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(cleaned[start:end+1])
                return result
            except json.JSONDecodeError: pass
        return {}

    def _fallback(self, sa: dict, sb: dict, err: str) -> Dict[str, Any]:
        def basic(persona, pname, focus):
            return {"persona": persona, "persona_name": pname, "focus_area": focus,
                    "attack_points": [f"LLM不可用, 无法详细评审 ({err[:80]})"],
                    "severity": "minor", "agreement": 0.5, "suggested_fixes": ["人工审查"],
                    "reference_to_report": "N/A"}
        return {"attacks_on_a": [basic("medchem","药化老兵","ADMET/类药性"),
                                 basic("funnel","漏斗终结者","存活率/效率"),
                                 basic("target","靶点专家","靶点适配性")],
                "attacks_on_b": [basic("medchem","药化老兵","ADMET/类药性"),
                                 basic("funnel","漏斗终结者","存活率/效率"),
                                 basic("target","靶点专家","靶点适配性")],
                "debate_summary": f"[FALLBACK] {err[:200]}"}


def red_team_debate_node(state: dict) -> dict:
    from datetime import datetime as dt
    ts = state.get("tournament_state", {})
    pairings = list(ts.get("pairings_queue", []))
    if not pairings:
        return {"pipeline_stage": "tournament", "event_log": ["[RedTeam] No more pairings."]}
    pair = pairings.pop(0)
    strategies = {s["strategy_name"]: s for s in state.get("candidate_strategies", [])}
    sa, sb = strategies.get(pair[0], {}), strategies.get(pair[1], {})
    if not sa or not sb:
        return {"tournament_state": {**ts, "pairings_queue": pairings}}
    reviewer = RedTeamReviewer()
    result = reviewer.debate_strategies(sa, sb, state.get("target_profile"))
    return {
        "pipeline_stage": "tournament",
        "tournament_state": {**ts, "pairings_queue": pairings},
        "event_log": [f"[{dt.now(timezone.utc).isoformat()}] [RedTeam] {pair[0][:30]} vs {pair[1][:30]}"],
        "_current_debate_pair": pair, "_current_debate_result": result,
    }
