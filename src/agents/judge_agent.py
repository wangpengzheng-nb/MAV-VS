"""
AutoVS-Agent v3.0: 锦标赛裁判 + ELO积分系统
=============================================
裁判: 读取策略原文 + 三人设评审报告 → 配对裁决 (强制CoT推理)
ELO: 带平局的动态K因子 + 冷门惩罚
"""

from __future__ import annotations

import json, os, re, math
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# Judge Schema
# =============================================================================

class VerdictDetail(BaseModel):
    """裁决详细维度对比。"""
    dimension: str = Field(default="")
    winner: str = Field(default="tie", description="A / B / tie")
    reasoning: str = Field(default="")


class StrategyVerdict(BaseModel):
    strategy_a_score: float = Field(default=50.0, ge=0, le=100)
    strategy_b_score: float = Field(default=50.0, ge=0, le=100)
    winner: str = Field(default="tie", description="胜出策略名 or 'tie'")

    # CoT 推理链
    dimension_comparison: List[VerdictDetail] = Field(default_factory=list)
    key_differentiator: str = Field(default="", description="两个策略的核心差异")
    risk_assessment: str = Field(default="", description="风险权衡分析")
    judge_commentary: str = Field(default="", description="裁判综合点评")

    suggestions_a: List[str] = Field(default_factory=list)
    suggestions_b: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1, description="裁判置信度")


VerdictDetail.model_rebuild()
StrategyVerdict.model_rebuild()


# =============================================================================
# Judge System Prompt
# =============================================================================

JUDGE_SYSTEM_PROMPT = """\
你是虚拟筛选策略首席裁判。你的裁决决定数千万美元的筛选方向。

## 评审流程 (强制 Chain-of-Thought)

你必须按以下5步推理, 每一步都要输出:

### Step 0: 完整性检查 (新增!)
检查双方策略是否缺少关键步骤:
  □ 是否有库定义(来源+大小)?
  □ 是否有阳性对照/诱饵分子验证?
  □ 是否有ADMET预过滤?
  □ 是否有后端精确验证(对接/MD/FEP至少一种)?
  □ 是否有应急预案(存活<10时的回退)?
  □ 是否引用了调研报告中的具体PDB ID/IC50数据?
每缺一项, 对应策略的相应维度扣5-10分。在dimension_comparison中明确说明扣分原因。

### Step 1: 逐维度对比
对比三个评审维度, 结合评审官报告和你的独立判断:
- 漏斗工程: 谁的管道设计更合理? 谁缺的步骤更多?
- 需求匹配: 谁更精准地满足了靶点特征和用户约束? 谁忽略了用户要求?
- 产出质量: 谁更可能产出高质量、多样化的hit分子?

每个维度明确给出胜者(A/B/tie)和理由。

### Step 2: 关键差异识别
两个策略最核心的区别是什么? 这个区别为什么决定胜负?

### Step 3: 风险权衡
A的风险是什么? B的风险是什么? 在当前靶点和用户需求的背景下,
哪个策略的风险更可控?

### Step 4: 最终裁决
综合评分 (0-100) + 胜者判定。
评分必须有区分度: 如果两个策略质量差距明显, 分数差应>10分。

## 裁决原则
1. 有critical_flaws的策略 → 即使其他方面优秀, 也应判负
2. 用户明确要求的功能 → 完全忽略的一方严重扣分
3. 内容过薄的策略(<3步且无ADMET/验证) → 不应超过60分
4. 平局是合法的裁决结果 → 当两个策略质量非常接近时(分差<3), 给tie
5. 引用评审官的发现, 但你的判断是独立的
6. confidence值: 分差<5→0.5-0.6, 5-15→0.7-0.8, >15→0.85-0.95

## 输出JSON格式
{
  "strategy_a_score": 75,
  "strategy_b_score": 68,
  "winner": "策略A的名称",
  "dimension_comparison": [
    {"dimension": "漏斗工程", "winner": "A", "reasoning": "A的管道...而B的管道..."},
    {"dimension": "需求匹配", "winner": "B", "reasoning": "B在...方面更精准..."},
    {"dimension": "产出质量", "winner": "tie", "reasoning": "两者各有优势..."}
  ],
  "key_differentiator": "核心差异描述",
  "risk_assessment": "风险分析",
  "judge_commentary": "综述",
  "suggestions_a": ["改进建议"],
  "suggestions_b": ["改进建议"],
  "confidence": 0.8
}
"""


# =============================================================================
# StrategyJudge
# =============================================================================

class StrategyJudge:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None,
                 temperature=0.1, max_tokens=4096):
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

    # =========================================================================
    # 主入口: 裁决一场配对
    # =========================================================================

    def judge_match(self, strategy_a: dict, strategy_b: dict,
                    review_a: dict, review_b: dict,
                    research_report: dict = None,
                    user_query: str = "") -> Dict[str, Any]:
        """裁决一场策略配对。

        Args:
            strategy_a/b: 策略原文
            review_a/b: TournamentReviewer 的三官评审结果
            research_report: 调研报告
            user_query: 用户原始任务

        Returns: StrategyVerdict dict
        """
        prompt = self._build_judge_prompt(
            strategy_a, strategy_b, review_a, review_b,
            research_report, user_query
        )

        try:
            is_reasoner = "reasoner" in self.model.lower()
            kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                          messages=[{"role":"system","content":JUDGE_SYSTEM_PROMPT},
                                    {"role":"user","content":prompt}])
            if not is_reasoner:
                kwargs["temperature"] = self.temperature
                kwargs["response_format"] = {"type":"json_object"}

            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = self._robust_json_parse(raw)

            verdict = StrategyVerdict.model_validate(parsed)
            result = verdict.model_dump()

            # 胜者名校验
            name_a = strategy_a.get("strategy_name", "")
            name_b = strategy_b.get("strategy_name", "")
            result = self._validate_winner(result, name_a, name_b)

            # 如果有critical_flaws, 降分
            flaws_a = review_a.get("all_critical_flaws", [])
            flaws_b = review_b.get("all_critical_flaws", [])
            if flaws_a:
                result["strategy_a_score"] = min(result["strategy_a_score"], 45)
            if flaws_b:
                result["strategy_b_score"] = min(result["strategy_b_score"], 45)

            return result

        except Exception as e:
            print(f"  ⚠️ 裁判调用失败: {e}, 使用启发式fallback", flush=True)
            return self._fallback_judge(strategy_a, strategy_b, review_a, review_b, str(e))

    # =========================================================================
    # Prompt 构建
    # =========================================================================

    def _build_judge_prompt(self, sa, sb, ra, rb, report, query):
        parts = []

        if query:
            parts.append(f"## 用户原始任务\n{query}\n")

        if report:
            parts.append(f"## 靶点背景\n"
                         f"靶点: {report.get('target_name','?')} | "
                         f"基因: {report.get('gene_symbol','?')} | "
                         f"物种: {report.get('target_organism','?')}\n")

        # 策略A
        parts.append(f"## 策略A: {sa.get('strategy_name','?')}")
        parts.append(f"方法: {sa.get('approach_type','?')}")
        parts.append(f"标签: {sa.get('strategy_tagline','?')}")
        parts.append(f"原理: {sa.get('rationale','?')[:400]}")
        parts.append(self._fmt_review_for_judge(ra, "A"))

        # 策略B
        parts.append(f"## 策略B: {sb.get('strategy_name','?')}")
        parts.append(f"方法: {sb.get('approach_type','?')}")
        parts.append(f"标签: {sb.get('strategy_tagline','?')}")
        parts.append(f"原理: {sb.get('rationale','?')[:400]}")
        parts.append(self._fmt_review_for_judge(rb, "B"))

        parts.append("请按4步推理链裁决。注意: 策略名较长时, winner字段必须填完整的策略名。")

        return "\n\n".join(parts)

    @staticmethod
    def _fmt_review_for_judge(review: dict, label: str) -> str:
        lines = [f"\n### 评审官对{label}的评价:"]
        for r in review.get("reports", []):
            lines.append(f"\n**{r.get('reviewer_name','?')}** "
                        f"总分: {r.get('overall_score','?'):.0f}/100")
            if r.get("critical_flaws"):
                lines.append(f"⚠️ 致命缺陷: {r['critical_flaws']}")
            for d in r.get("dimension_scores", []):
                lines.append(f"  - {d.get('name','?')}: {d.get('score','?'):.0f}分 "
                            f"({d.get('comment','?')[:100]})")
            if r.get("key_strengths"):
                lines.append(f"  优势: {'; '.join(r['key_strengths'][:3])}")
            if r.get("key_weaknesses"):
                lines.append(f"  劣势: {'; '.join(r['key_weaknesses'][:3])}")
        lines.append(f"\n{label}加权综合分: {review.get('weighted_score','?'):.1f}")
        if review.get("all_critical_flaws"):
            lines.append(f"⚠️ {label}全部致命缺陷: {review['all_critical_flaws']}")
        return "\n".join(lines)

    # =========================================================================
    # Fallback
    # =========================================================================

    def _fallback_judge(self, sa, sb, ra, rb, err):
        """启发式裁决: 基于评审官加权综合分。"""
        score_a = ra.get("weighted_score", 50)
        score_b = rb.get("weighted_score", 50)
        flaws_a = len(ra.get("all_critical_flaws", []))
        flaws_b = len(rb.get("all_critical_flaws", []))

        # 致命缺陷惩罚
        if flaws_a:
            score_a -= 20
        if flaws_b:
            score_b -= 20

        name_a = sa.get("strategy_name", "A")
        name_b = sb.get("strategy_name", "B")

        winner = name_a if score_a > score_b else (name_b if score_b > score_a else "tie")
        diff = abs(score_a - score_b)
        confidence = 0.5 if diff < 5 else (0.7 if diff < 15 else 0.85)

        return {
            "strategy_a_score": round(max(0, score_a), 1),
            "strategy_b_score": round(max(0, score_b), 1),
            "winner": winner,
            "dimension_comparison": [],
            "key_differentiator": f"启发式裁决(LLM不可用: {err[:80]})",
            "risk_assessment": "N/A",
            "judge_commentary": f"三官加权分: A={score_a:.1f}, B={score_b:.1f}",
            "suggestions_a": [],
            "suggestions_b": [],
            "confidence": confidence,
        }

    # =========================================================================
    # ELO 积分系统
    # =========================================================================

    @staticmethod
    def compute_initial_elo(review_result: dict, base: int = 1000) -> float:
        """从独立评审结果计算初始ELO。

        公式: initial_elo = base + weighted_score × 10
        范围: 1000 (0分) ~ 2000 (100分)
        """
        ws = review_result.get("weighted_score", 50)
        return base + ws * 10

    @staticmethod
    def compute_k_factor(elo_a: float, elo_b: float) -> float:
        """动态K因子: 基于ELO差距。

        差距 < 50  → K=24 (实力相近, 小幅调整)
        差距 50-150 → K=32 (正常差距)
        差距 > 150  → K=48 (冷门/碾压, 大幅调整)
        """
        diff = abs(elo_a - elo_b)
        if diff < 50:
            return 24.0
        elif diff < 150:
            return 32.0
        else:
            return 48.0

    @staticmethod
    def update_elo(elo_ratings: dict, winner: str, loser: str,
                   is_tie: bool = False) -> Tuple[float, float, dict]:
        """更新ELO积分。

        Args:
            elo_ratings: {name: elo} dict (会被原地修改)
            winner: 胜者名 (tie时任意一方)
            loser: 败者名 (tie时任意另一方)
            is_tie: 是否平局

        Returns:
            (shift_a, shift_b, updated_ratings)
        """
        ra = elo_ratings.get(winner, 1500.0)
        rb = elo_ratings.get(loser, 1500.0)

        K = StrategyJudge.compute_k_factor(ra, rb)

        if is_tie:
            # 平局: S_A = S_B = 0.5
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            eb = 1.0 / (1.0 + 10.0 ** ((ra - rb) / 400.0))
            shift_a = K * (0.5 - ea)
            shift_b = K * (0.5 - eb)
            # 高分方微扣, 低分方微加
        else:
            # 胜者: S=1, 败者: S=0
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            shift_win = K * (1.0 - ea)

            # 冷门惩罚: 高分输给低分 → K × 1.5
            if ra > rb + 50:
                # 胜者是低分方(冷门), 对高分方额外惩罚
                shift_lose = -K * 0.5 * 1.5  # 高分方多扣
            else:
                shift_lose = -shift_win

            if winner == list(elo_ratings.keys())[list(elo_ratings.values()).index(ra)] if ra in elo_ratings.values() else False:
                pass  # 保持winner匹配

            # 确定转移方向
            if elo_ratings.get(winner, 1500) == ra:
                # winner 是elO_a (高分方)
                shift_a = shift_win
                shift_b = -shift_win
                if ra > rb + 50:
                    shift_b *= 1.5  # 高分输低分, 多扣
                    shift_a = -shift_b  # 维持零和
            else:
                # winner 是低分方
                shift_a = shift_win
                shift_b = -shift_win
                if rb > ra + 50:
                    shift_b *= 1.5
                    shift_a = -shift_b

        # 应用变化
        new_ratings = dict(elo_ratings)
        ra_new = ra
        rb_new = rb

        if is_tie:
            # E_A = 1/(1+10^((rb-ra)/400))
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            shift = K * (0.5 - ea)
            ra_new = ra + shift
            rb_new = rb - shift
            shift_a = shift
            shift_b = -shift
        else:
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            gain = K * (1.0 - ea)

            # 冷门惩罚
            upset_penalty = 1.0
            if ra > rb:  # winner是高分方, 正常
                ra_new = ra + gain
                rb_new = rb - gain
                shift_a = gain
                shift_b = -gain
            else:  # winner是低分方(冷门)
                if rb - ra > 50:
                    upset_penalty = 1.5
                ra_new = ra + gain * upset_penalty
                rb_new = rb - gain * upset_penalty
                shift_a = gain * upset_penalty
                shift_b = -gain * upset_penalty

        new_ratings[winner] = ra_new
        new_ratings[loser] = rb_new

        return shift_a, shift_b, new_ratings

    # =========================================================================
    # 辅助方法
    # =========================================================================

    @staticmethod
    def _validate_winner(verdict: dict, name_a: str, name_b: str) -> dict:
        """校验LLM返回的winner是否为实际策略名。"""
        winner = str(verdict.get("winner", "")).strip()
        if winner == name_a or winner == name_b:
            return verdict
        if winner.lower() in ("strategy_a", "strategy a", "a"):
            verdict["winner"] = name_a; return verdict
        if winner.lower() in ("strategy_b", "strategy b", "b"):
            verdict["winner"] = name_b; return verdict
        if name_a in winner:
            verdict["winner"] = name_a; return verdict
        if name_b in winner:
            verdict["winner"] = name_b; return verdict
        # 按分数裁决
        sa = verdict.get("strategy_a_score", 50)
        sb = verdict.get("strategy_b_score", 50)
        verdict["winner"] = name_a if sa > sb else (name_b if sb > sa else "tie")
        return verdict

    @staticmethod
    def _robust_json_parse(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.JSONDecoder().raw_decode(cleaned[s:e+1])[0]
            except json.JSONDecodeError:
                pass
        return {}

    # =========================================================================
    # 兼容旧接口
    # =========================================================================

    def judge_debate(self, strategy_a, strategy_b, attacks_on_a, attacks_on_b,
                     target_profile=None):
        """兼容旧版接口: 将旧式attacks转为新式review。"""
        # 从attacks提取分数
        def _score_from_attacks(attacks):
            if not attacks:
                return 50
            avg_agree = sum(a.get("agreement", 0.5) for a in attacks) / len(attacks)
            crit_count = sum(1 for a in attacks if a.get("severity") == "critical")
            return max(10, round(avg_agree * 100 - crit_count * 20))

        fake_review_a = {
            "weighted_score": _score_from_attacks(attacks_on_a),
            "reports": [],
            "all_critical_flaws": [],
        }
        fake_review_b = {
            "weighted_score": _score_from_attacks(attacks_on_b),
            "reports": [],
            "all_critical_flaws": [],
        }

        return self.judge_match(strategy_a, strategy_b,
                                fake_review_a, fake_review_b,
                                target_profile)


# =============================================================================
# 兼容: LangGraph node
# =============================================================================

def judge_node(state: dict) -> dict:
    from datetime import datetime as dt
    review_a = state.get("_current_review_a", {})
    review_b = state.get("_current_review_b", {})
    pair = state.get("_current_debate_pair", ["", ""])
    if not review_a or not pair[0]:
        return {"event_log": ["[Judge] No reviews to judge."]}

    strategies = {s["strategy_name"]: s for s in state.get("candidate_strategies", [])}
    sa = strategies.get(pair[0], {})
    sb = strategies.get(pair[1], {})

    judge = StrategyJudge()
    verdict = judge.judge_match(sa, sb, review_a, review_b,
                                state.get("target_profile"))
    elo = dict(state.get("tournament_state", {}).get("elo_ratings", {}))

    winner_name = verdict.get("winner", "")
    is_tie = (winner_name == "tie")
    if winner_name and winner_name != "tie":
        loser_name = pair[1] if winner_name == pair[0] else pair[0]
        gain, loss, elo = StrategyJudge.update_elo(elo, winner_name, loser_name, is_tie)

    leader = max(elo.items(), key=lambda x: x[1])[0] if elo else ""
    now = dt.now().isoformat()

    return {
        "pipeline_stage": "tournament",
        "tournament_state": {**state["tournament_state"],
            "elo_ratings": elo,
            "completed_debates": state["tournament_state"]["completed_debates"] + 1,
            "current_leader": leader,
            "round_number": state["tournament_state"]["round_number"] + 1,
        },
        "tournament_history": state.get("tournament_history", []) + [{
            "round_id": f"round_{state['tournament_state']['completed_debates']+1}",
            "strategy_a": pair[0], "strategy_b": pair[1],
            "judge_summary": verdict.get("judge_commentary", ""),
            "winner": winner_name,
            "elo_shift_a": gain if 'gain' in dir() else 0,
            "elo_shift_b": loss if 'loss' in dir() else 0,
            "confidence": verdict.get("confidence", 0.5),
            "timestamp": now,
        }],
        "_current_debate_pair": [], "_current_debate_result": {},
        "_current_review_a": {}, "_current_review_b": {},
        "updated_at": now,
        "event_log": [f"[{now}] [Judge] {pair[0][:30]} vs {pair[1][:30]}: "
                      f"Winner={winner_name or 'tie'} "
                      f"(conf={verdict.get('confidence',0.5):.0%})"],
    }
