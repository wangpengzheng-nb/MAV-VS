"""
AutoVS-Agent v4: 策略排名评审 — 三人设 pairwise 投票
========================================================
三位评审官, 对策略两两对比, 各投一票(胜负平), 输出结构化诊断建议。
"""
from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ═══════════════════════════════════════════════════
# Reviewer Prompts (v4: pairwise voting)
# ═══════════════════════════════════════════════════

REVIEWER_SCIENCE_PROMPT = """\
你是「科学合理性与药化评审官」。

你的任务: 比较策略A和策略B, 从3个维度判断哪个更优秀。

## 维度1: 靶点适配性
- 方法是否与靶点类型+口袋特征匹配? (PPI用flat_ppi方法, 激酶用hinge-binding)
- 关键残基是否正确引用? action_type选择是否合理?
- N/A条件: 两个策略都没有明确靶点信息

## 维度2: 药化/设计合理性
- 化学可行性、合成可及性、类药性、PAINS排除
- pipeline中的参数阈值是否合理? 是否有明显的药化规则违反?
- N/A条件: 两个策略都只有概念层描述, 无具体参数

## 维度3: 用户约束与先验知识符合度
- 用户任务中的"不要X/禁止X/避开X"是否被策略满足?
- 用户任务中的"必须Y/要求Y"是否被策略包含?
- 先验知识(如有)是否被遵守? (如工具选择规则)
- N/A条件: 无用户约束也无先验知识

## 综合投票优先级
1. 靶点适配性判定为不可行 → 直接判负
2. 用户约束有致命违反 → 直接判负
3. 两项科学维度均打平 → 综合判平

## 输出格式 (纯JSON)
{
  "reviewer_id": "science_chemistry",
  "dimension_votes": [
    {"dimension": "靶点适配性", "winner": "A", "reasoning_a": "...", "reasoning_b": "..."},
    {"dimension": "药化设计合理性", "winner": "B", "reasoning_a": "...", "reasoning_b": "..."},
    {"dimension": "用户约束符合度", "winner": "tie", "reasoning_a": "...", "reasoning_b": "..."}
  ],
  "overall_verdict": "A",
  "decision_logic": "核心权衡理由(50-100字)",
  "verdict_confidence": "high",
  "critical_concerns": {
    "A": [{"step_id": "uuid-x", "action_type": "...", "issue": "...", "consequence": "...", "severity": "Warning"}],
    "B": []
  },
  "suggestions": {
    "A": [{"step_id": "uuid-x", "action": "UPDATE_PARAM uuid-x mw_range [400,800]", "priority": "High", "feasibility": "Easy", "rationale": "..."}],
    "B": []
  }
}

注意: step_id 必须使用策略中定义的UUID, 不要用step_number。suggestions的action字段使用DSL指令。
"""

REVIEWER_ENGINEERING_PROMPT = """\
你是「工程可执行性评审官」。

你的任务: 比较策略A和策略B, 从3个维度判断哪个更可执行。

## 维度1: Action链完整性与逻辑
- 每个action的input/output是否清晰匹配? 是否存在断链?
- pipeline的逻辑顺序是否合理? 是否有冗余步骤?
- N/A条件: 两个策略都只有1步, 无法对比

## 维度2: 工具与资源可用性
- action所需工具是否开源/可获取? (优先开源)
- computational_cost估算是否合理?
- N/A条件: 两个策略都未定义computational_cost

## 维度3: 鲁棒性与扩展性
- 参数敏感度如何? 失败后是否有contingency?
- pipeline能否处理更大的库?
- N/A条件: 两个策略都只有基础描述, 无参数细节

## 综合投票优先级
1. Action链有致命漏洞(必断链) → 直接判负
2. 完整性合格 → 优先看工具可用性
3. 两者都可用 → 看鲁棒性

## 输出格式 (纯JSON)
{
  "reviewer_id": "engineering",
  "dimension_votes": [...],
  "overall_verdict": "A",
  "decision_logic": "...",
  "verdict_confidence": "high",
  "critical_concerns": {...},
  "suggestions": {...}
}

注意: step_id必须使用策略中定义的UUID。suggestions的action字段使用DSL指令(UPDATE_PARAM/INSERT_STEP/REMOVE_STEP/REPLACE_ACTION/ADD_PARAM)。
"""

REVIEWER_RISK_PROMPT = """\
你是「风险与创新性评审官」。

你的任务: 比较策略A和策略B, 从3个维度判断哪个风险更低、创新性更好。

## 维度1: 化学空间与多样性
- 是否有diversity_selection? 化学空间覆盖够广吗?
- 骨架单一风险? 是否只聚焦已知配体空间?
- N/A条件: 两个策略都无多样性步骤

## 维度2: 假阳性/假阴性风险
- 筛选逻辑是否会产生大量假阳性或漏掉真阳性?
- quality_criteria是否明确? 是否有validation步骤?
- N/A条件: 两个策略都只有1-2步筛选, 无法评估

## 维度3: 新颖性与可解释性
- 有机会发现新骨架/新机制吗? 还是只在已知空间打转?
- 策略逻辑是否清晰可审计? 每步判断标准是否明确?
- N/A条件: 两个策略都无新颖性相关内容

## 综合投票优先级
1. 假阳性/假阴性风险极高 → 判负
2. 风险相近 → 优先选化学空间覆盖更广的
3. 都差不多 → 看新颖性

## 输出格式 (纯JSON)
{
  "reviewer_id": "risk_innovation",
  "dimension_votes": [...],
  "overall_verdict": "A",
  "decision_logic": "...",
  "verdict_confidence": "high",
  "critical_concerns": {...},
  "suggestions": {...}
}

注意: step_id必须使用策略中定义的UUID。suggestions的action字段使用DSL指令。
"""

REVIEWER_CONFIGS = [
    {"id": "science_chemistry", "name": "科学合理性与药化评审官", "prompt": REVIEWER_SCIENCE_PROMPT},
    {"id": "engineering", "name": "工程可执行性评审官", "prompt": REVIEWER_ENGINEERING_PROMPT},
    {"id": "risk_innovation", "name": "风险与创新性评审官", "prompt": REVIEWER_RISK_PROMPT},
]

# ═══════════════════════════════════════════════════
# TournamentReviewer
# ═══════════════════════════════════════════════════

class TournamentReviewer:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None,
                 temperature=0.15, max_tokens=4096):
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

    # ═══════════════════════════════════════════════════
    # v4: pairwise 对比 + 投票
    # ═══════════════════════════════════════════════════

    def compare_strategies(self, strategy_a: dict, strategy_b: dict,
                           research_report: dict, user_query: str = "",
                           prior_knowledge: str = "", reviewer_id: str = None,
                           match_id: str = "") -> Dict[str, Any]:
        """单评审官对一对策略进行对比投票。"""
        cfg = next((c for c in REVIEWER_CONFIGS if c["id"] == reviewer_id), REVIEWER_CONFIGS[0])
        context = self._build_compare_context(strategy_a, strategy_b, research_report,
                                               user_query, prior_knowledge)

        try:
            is_reasoner = "reasoner" in self.model.lower()
            msgs = [
                {"role": "system", "content": cfg["prompt"]},
                {"role": "user", "content": context},
                {"role": "system", "content": f"请以{cfg['name']}身份比较策略A和B, 输出完整JSON。"},
            ]
            kwargs = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs)
            if not is_reasoner:
                kwargs["temperature"] = self.temperature
                kwargs["response_format"] = {"type": "json_object"}

            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = self._robust_parse(raw)

            # 强制覆盖: 防LLM幻觉写错ID
            parsed["reviewer_id"] = reviewer_id or cfg["id"]
            parsed["match_id"] = match_id
            parsed.setdefault("overall_verdict", "tie")
            parsed.setdefault("verdict_confidence", "medium")
            parsed.setdefault("dimension_votes", [])
            parsed.setdefault("critical_concerns", {})
            parsed.setdefault("suggestions", {})
            parsed.setdefault("decision_logic", "")

            return parsed

        except Exception as e:
            return {
                "reviewer_id": reviewer_id or cfg["id"],
                "match_id": match_id,
                "dimension_votes": [],
                "overall_verdict": "tie",
                "decision_logic": f"LLM调用失败: {str(e)[:100]}",
                "verdict_confidence": "low",
                "critical_concerns": {},
                "suggestions": {},
            }

    def _build_compare_context(self, sa: dict, sb: dict, report: dict,
                                user_query: str, prior_knowledge: str) -> str:
        parts = []
        if user_query:
            parts.append(f"## 用户任务\n{user_query}\n")
        if prior_knowledge and prior_knowledge.strip():
            parts.append(f"## 先验知识\n{prior_knowledge.strip()}\n")
        parts.append(f"## 靶点背景\n靶点: {report.get('target_name','?')} | "
                     f"基因: {report.get('gene_symbol','?')}\n")
        parts.append(self._fmt_one("策略A", sa))
        parts.append(self._fmt_one("策略B", sb))
        return "\n\n".join(parts)

    @staticmethod
    def _fmt_one(label: str, s: dict) -> str:
        steps = s.get("pipeline", s.get("pipeline_steps", []))
        steps_text = ""
        for st in steps:
            sid = st.get("step_id", f"step_{st.get('step_number','?')}")
            atype = st.get("action_type", st.get("tool", "?"))
            aname = st.get("action_name", st.get("step_name", "?"))
            desc = st.get("description", st.get("action", ""))
            params = st.get("parameters", {})
            cost = st.get("computational_cost", "?")
            steps_text += (f"  [{sid}] {aname} ({atype}) cost={cost}\n"
                           f"    desc: {desc[:150]}\n"
                           f"    params: {json.dumps(params, ensure_ascii=False)[:100]}\n")
        approach = s.get("approach_category", s.get("approach_type", "?"))
        return f"""## {label}: {s.get('strategy_name','?')}
方法: {approach}
标签: {s.get('strategy_tagline','?')}
原理: {(s.get('rationale','?') or '')[:300]}
pipeline ({len(steps)}步):
{steps_text}
存活: {s.get('survival_estimate','?')}
应急: {str(s.get('contingency_plan', s.get('contingency','?')))[:150]}"""

    # ═══════════════════════════════════════════════════
    # 向后兼容: 旧版 review_strategy 接口
    # ═══════════════════════════════════════════════════

    def review_strategy(self, strategy: dict, research_report: dict,
                        user_query: str = "", prior_knowledge: str = "") -> Dict[str, Any]:
        """旧版兼容接口: 返回一个占位dict, 实际评审请用 compare_strategies"""
        return {"strategy_name": strategy.get("strategy_name", "?"),
                "weighted_score": 50.0, "reports": [], "all_critical_flaws": []}

    def calibrate_all(self, review_results, strategies, report, query):
        """旧版兼容: no-op"""
        return review_results

    # ═══════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _robust_parse(raw: str) -> Dict[str, Any]:
        try: return json.loads(raw)
        except json.JSONDecodeError: pass
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        try: return json.loads(cleaned)
        except json.JSONDecodeError: pass
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            try: return json.JSONDecoder().raw_decode(cleaned[s:e+1])[0]
            except json.JSONDecodeError: pass
        return {}

# 旧版别名
RedTeamReviewer = TournamentReviewer


def red_team_debate_node(state: dict) -> dict:
    return {"event_log": ["[Tournament] v4 pairwise voting — use compare_strategies"]}
