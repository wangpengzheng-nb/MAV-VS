"""
AutoVS-Agent v2.0: Judge Agent — 策略锦标赛裁判
==================================================
职责: 读取红军的辩论记录，对两个候选策略进行打分,
     更新 Elo 积分，选出一轮辩论的胜者。

与 v1 差异:
  旧: 对分子进行 1v1 辩论 + Elo (分子级)
  新: 对策略进行评审打分 + Elo (策略级)
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. Pydantic Schema: 策略裁判裁决
# =============================================================================

class StrategyVerdict(BaseModel):
    """裁判对一场策略辩论的裁决。"""

    strategy_a_score: float = Field(
        ..., ge=0, le=100,
        description="策略 A 的综合评分 (0-100)。基于红军攻击的严重程度和策略的适应性。"
    )
    strategy_b_score: float = Field(..., ge=0, le=100)
    winner: str = Field(..., description="胜出策略名 or 'tie'")
    key_deciding_factor: str = Field(
        ..., description="决定性因素 (50-150字)。明确指出为什么一方胜出。"
    )
    judge_commentary: str = Field(
        ..., description="裁判综合点评 (150-300字)。综合三位红军专家的意见给出最终判断。"
    )
    suggestions_for_loser: List[str] = Field(
        ..., min_length=1, description="对败方策略的具体改进建议"
    )


StrategyVerdict.model_rebuild()


# =============================================================================
# 2. 裁判系统提示词
# =============================================================================

JUDGE_SYSTEM_PROMPT = """\
# 人设: 虚拟筛选策略首席裁判 (Chief VS Strategy Judge)

你是一位在药物发现计算领域有 20 年经验的**首席裁判**。
你的职责是审阅红军评审团对两个虚拟筛选策略的攻击意见,
综合判断哪个策略更优秀, 并给出最终分数和排名。

---

# 评判维度 (权重)

## 1. 科学合理性 (40%)
- 策略的方法学是否适合该靶点?
- 过滤条件是否有充分的科学依据?
- 是否区分了绝对过滤、相对排序和软指标?

## 2. 可执行性 (30%)
- 策略是否可以在实际虚拟筛选管道中实现?
- 存活率估算是否合理?
- 应急预案是否充分?

## 3. 创新性与靶点适配 (20%)
- 策略是否体现了对靶点特性的深刻理解?
- 是否避免了"一刀切"的通用策略?

## 4. 红军攻击严重度 (10%)
- 红军是否发现了致命缺陷 (critical)?
- 这些缺陷是否可以修复?

---

# 评分规则
1. 分数范围: 0-100 分
2. 如果一方有 critical 级别的红军攻击且无法修复 → 该方分数应 < 50
3. 如果双方质量接近 → 分数接近, 但必须选出一方 (或 tie)
4. 对于靶点类型完全不匹配的策略 → 最高不超过 40 分

---

# 输出格式
严格输出 StrategyVerdict JSON Schema。
"""


# =============================================================================
# 3. StrategyJudge
# =============================================================================

class StrategyJudge:
    """策略锦标赛裁判 — 读取辩论记录 + 评分 + Elo 更新。"""

    def __init__(
        self,
        model: str = "deepseek-reasoner",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
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

    def judge_debate(
        self,
        strategy_a: dict,
        strategy_b: dict,
        attacks_on_a: List[dict],
        attacks_on_b: List[dict],
        target_profile: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """对一场辩论进行裁判。

        Returns:
            StrategyVerdict dict + computed Elo shifts
        """
        user_prompt = self._build_judge_prompt(
            strategy_a, strategy_b, attacks_on_a, attacks_on_b, target_profile,
        )

        try:
            is_reasoner = "reasoner" in self.model.lower()
            kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                          messages=[{"role":"system","content":JUDGE_SYSTEM_PROMPT},
                                    {"role":"user","content":user_prompt},
                                    {"role":"system","content":f"必须输出纯JSON:\n{json.dumps(StrategyVerdict.model_json_schema(),indent=2,ensure_ascii=False)}"}])
            if not is_reasoner: kwargs.update(temperature=self.temperature, response_format={"type":"json_object"})
            response = self.client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            if not raw or not raw.strip():
                raw = getattr(response.choices[0].message, "reasoning_content", "") or ""
                if "{" in raw: raw = raw[raw.rfind("{"):]
            raw = raw.strip()
            parsed = json.loads(raw)
            verdict = StrategyVerdict.model_validate(parsed)

            # ---- 胜者名校验: 防止LLM返回 "strategy_a" / "策略A" 等标签 ----
            name_a = strategy_a.get("strategy_name", "")
            name_b = strategy_b.get("strategy_name", "")
            verdict_dict = verdict.model_dump()
            verdict_dict = StrategyJudge._validate_winner(
                verdict_dict, name_a, name_b,
            )
            return verdict_dict
        except Exception as e:
            return self._fallback_verdict(strategy_a, strategy_b, attacks_on_a, attacks_on_b, str(e))

    def _build_judge_prompt(
        self, sa: dict, sb: dict,
        attacks_a: List[dict], attacks_b: List[dict],
        profile: Optional[dict],
    ) -> str:
        def fmt_attacks(attacks: List[dict], label: str) -> str:
            lines = [f"\n### 红军对 {label} 的攻击:"]
            for atk in attacks:
                lines.append(f"- [{atk.get('persona_name', '?')}] 严重度={atk.get('severity', '?')} | 认可度={atk.get('agreement_with_strategy', 0):.1%}")
                for p in atk.get("attack_points", []):
                    lines.append(f"  • {p}")
                for f in atk.get("suggested_fixes", []):
                    lines.append(f"  → 建议: {f}")
            return "\n".join(lines)

        return f"""\
## 策略辩论裁判任务

### 策略 A: {sa.get('strategy_name', '?')}
- 方法: {sa.get('approach_type', '?')}
- 标签: {sa.get('strategy_tagline', '')}
{fmt_attacks(attacks_a, '策略A')}

### 策略 B: {sb.get('strategy_name', '?')}
- 方法: {sb.get('approach_type', '?')}
- 标签: {sb.get('strategy_tagline', '')}
{fmt_attacks(attacks_b, '策略B')}

### 请裁决
"""

    def _fallback_verdict(
        self, sa: dict, sb: dict,
        attacks_a: List[dict], attacks_b: List[dict],
        error: str,
    ) -> Dict[str, Any]:
        """启发式降级裁决: 基于红军认可度平均值。"""
        def avg_agreement(attacks: List[dict]) -> float:
            vals = [a.get("agreement_with_strategy", 0.5) for a in attacks]
            return sum(vals) / len(vals) if vals else 0.5

        def count_critical(attacks: List[dict]) -> int:
            return sum(1 for a in attacks if a.get("severity") == "critical")

        avg_a = avg_agreement(attacks_a) - count_critical(attacks_a) * 0.15
        avg_b = avg_agreement(attacks_b) - count_critical(attacks_b) * 0.15

        score_a = round(avg_a * 100)
        score_b = round(avg_b * 100)
        winner = sa["strategy_name"] if score_a > score_b else sb["strategy_name"] if score_b > score_a else "tie"

        return {
            "strategy_a_score": score_a,
            "strategy_b_score": score_b,
            "winner": winner,
            "key_deciding_factor": f"启发式裁决 (LLM不可用: {error[:80]})",
            "judge_commentary": f"基于红军认可度平均值: A={avg_a:.2f}, B={avg_b:.2f}",
            "suggestions_for_loser": ["请人工审查辩论记录"],
        }

    @staticmethod
    def _validate_winner(
        verdict: Dict[str, Any],
        name_a: str,
        name_b: str,
    ) -> Dict[str, Any]:
        """校验 LLM 返回的 winner 是否是实际策略名。

        LLM 可能返回 "strategy_a" / "策略A" / "A" / 策略名本身。
        此方法将非实际策略名的返回值映射为正确的名称。
        """
        winner = str(verdict.get("winner", "")).strip()

        # 直接匹配成功
        if winner == name_a or winner == name_b:
            return verdict

        # LLM 返回了 "strategy_a" 或 "strategy_A"
        if winner.lower() in ("strategy_a", "strategy a", "a"):
            verdict["winner"] = name_a
            verdict["judge_commentary"] += f" [winner corrected: '{winner}' → '{name_a}']"
            return verdict

        # LLM 返回了 "strategy_b" 或 "strategy_B"
        if winner.lower() in ("strategy_b", "strategy b", "b"):
            verdict["winner"] = name_b
            verdict["judge_commentary"] += f" [winner corrected: '{winner}' → '{name_b}']"
            return verdict

        # LLM 返回了 "策略A" / "保守SBDD" 等包含关键词的
        if name_a in winner or (len(winner) < 3 and winner.upper() == "A"):
            verdict["winner"] = name_a
            return verdict
        if name_b in winner or (len(winner) < 3 and winner.upper() == "B"):
            verdict["winner"] = name_b
            return verdict

        # 兜底: 启发式裁决
        score_a = verdict.get("strategy_a_score", 50)
        score_b = verdict.get("strategy_b_score", 50)
        if score_a > score_b:
            verdict["winner"] = name_a
        elif score_b > score_a:
            verdict["winner"] = name_b
        else:
            verdict["winner"] = "tie"
        verdict["judge_commentary"] += (
            f" [winner auto-resolved from scores: "
            f"A={score_a}, B={score_b}, → {verdict['winner']}]"
        )
        return verdict

    @staticmethod
    def update_elo(
        elo_ratings: Dict[str, float],
        winner_name: str,
        loser_name: str,
        k_factor: float = 32.0,
    ) -> Tuple[float, float]:
        """更新 Elo 积分。"""
        ra = elo_ratings.get(winner_name, 1500.0)
        rb = elo_ratings.get(loser_name, 1500.0)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        shift = k_factor * (1.0 - ea)
        return shift, shift  # (winner_gain, loser_loss)


# =============================================================================
# LangGraph 节点
# =============================================================================

def judge_node(state: dict) -> dict:
    """锦标赛裁判节点: 读取当前辩论结果 → 裁决 → 更新 Elo → 记录。"""
    from datetime import datetime as dt

    debate_result = state.get("_current_debate_result", {})
    pair = state.get("_current_debate_pair", ["", ""])

    if not debate_result or not pair[0]:
        return {"event_log": ["[Judge] No debate to judge."]}

    strategies = {s["strategy_name"]: s for s in state.get("candidate_strategies", [])}
    sa = strategies.get(pair[0], {})
    sb = strategies.get(pair[1], {})
    elo = dict(state.get("tournament_state", {}).get("elo_ratings", {}))

    judge = StrategyJudge()
    verdict = judge.judge_debate(
        strategy_a=sa, strategy_b=sb,
        attacks_on_a=debate_result.get("attacks_on_a", []),
        attacks_on_b=debate_result.get("attacks_on_b", []),
        target_profile=state.get("target_profile"),
    )

    # Elo 更新
    winner_name = verdict.get("winner", "")
    loser_name = pair[1] if winner_name == pair[0] else pair[0] if winner_name == pair[1] else ""
    if winner_name and loser_name and winner_name != "tie":
        gain, loss = StrategyJudge.update_elo(elo, winner_name, loser_name, state["tournament_state"]["elo_k_factor"])
        elo[winner_name] = elo.get(winner_name, 1500.0) + gain
        elo[loser_name] = elo.get(loser_name, 1500.0) - loss

    # 当前领先者
    leader = max(elo.items(), key=lambda x: x[1])[0] if elo else ""

    # 记录本轮辩论
    record = {
        "round_id": f"round_{state['tournament_state']['completed_debates'] + 1}",
        "strategy_a": pair[0],
        "strategy_b": pair[1],
        "expert_attacks_on_a": debate_result.get("attacks_on_a", []),
        "expert_attacks_on_b": debate_result.get("attacks_on_b", []),
        "judge_summary": verdict.get("judge_commentary", ""),
        "winner": winner_name,
        "elo_shift_a": elo.get(pair[0], 1500.0) - state["tournament_state"]["elo_ratings"].get(pair[0], 1500.0),
        "elo_shift_b": elo.get(pair[1], 1500.0) - state["tournament_state"]["elo_ratings"].get(pair[1], 1500.0),
        "key_deciding_factor": verdict.get("key_deciding_factor", ""),
        "timestamp": dt.now(timezone.utc).isoformat(),
    }

    now = dt.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "tournament",
        "tournament_state": {
            **state["tournament_state"],
            "elo_ratings": elo,
            "completed_debates": state["tournament_state"]["completed_debates"] + 1,
            "current_leader": leader,
            "round_number": state["tournament_state"]["round_number"] + 1,
        },
        "tournament_history": state.get("tournament_history", []) + [record],
        "_current_debate_pair": [],
        "_current_debate_result": {},
        "updated_at": now,
        "event_log": [
            f"[{now}] [Judge] {pair[0]} vs {pair[1]}: "
            f"Winner={winner_name or 'tie'}, "
            f"Scores: A={verdict.get('strategy_a_score', '?')}, B={verdict.get('strategy_b_score', '?')}. "
            f"Leader: {leader} ({elo.get(leader, 0):.0f} Elo)"
        ],
    }
