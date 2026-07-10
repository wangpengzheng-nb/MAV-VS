"""
AutoVS-Agent v2.0: Judge Agent — 策略锦标赛裁判
==================================================
读取红军辩论记录, 对候选策略打分, 更新 Elo 积分。
"""

from __future__ import annotations

import json, os, re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field


class StrategyVerdict(BaseModel):
    strategy_a_score: float = Field(default=50.0, ge=0, le=100)
    strategy_b_score: float = Field(default=50.0, ge=0, le=100)
    winner: str = Field(default="tie", description="胜出策略名 or 'tie'")
    key_deciding_factor: str = Field(default="", description="决定性因素")
    judge_commentary: str = Field(default="", description="裁判综合点评")
    suggestions_for_loser: List[str] = Field(default_factory=list, description="改进建议")

StrategyVerdict.model_rebuild()

JUDGE_SYSTEM_PROMPT = """\
# 人设: 虚拟筛选策略首席裁判

你审阅红军评审团对两个虚拟筛选策略的攻击意见, 综合判断哪个策略更优秀。

评判维度: 科学合理性(40%) / 可执行性(30%) / 创新与靶点适配(20%) / 红军攻击严重度(10%)

分数范围: 0-100。如果一方有critical级别攻击且无法修复, 该方分数应<50。
"""


class StrategyJudge:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.1, max_tokens=2048):
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

    def judge_debate(self, strategy_a, strategy_b, attacks_on_a, attacks_on_b, target_profile=None):
        user_prompt = self._build_prompt(strategy_a, strategy_b, attacks_on_a, attacks_on_b, target_profile)
        is_reasoner = "reasoner" in self.model.lower()
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role":"system","content":JUDGE_SYSTEM_PROMPT},
                                {"role":"user","content":user_prompt},
                                {"role":"system","content":'输出JSON:{"strategy_a_score":50,"strategy_b_score":50,"winner":"策略名","key_deciding_factor":"...","judge_commentary":"...","suggestions_for_loser":["..."]}'}])
        if not is_reasoner:
            kwargs["temperature"] = self.temperature
            kwargs["response_format"] = {"type":"json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raw = getattr(response.choices[0].message, "reasoning_content", "") or ""
            if not raw.strip():
                return self._fallback(strategy_a, strategy_b, attacks_on_a, attacks_on_b, "Empty LLM response")
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = StrategyJudge._robust_json_parse(raw.strip())
            verdict = StrategyVerdict.model_validate(parsed)
            result = verdict.model_dump()
            # 🆕 胜者名校验
            result = StrategyJudge._validate_winner(
                result,
                strategy_a.get("strategy_name", ""),
                strategy_b.get("strategy_name", ""),
            )
            return result
        except Exception as e:
            return self._fallback(strategy_a, strategy_b, attacks_on_a, attacks_on_b, str(e))

    @staticmethod
    def _robust_json_parse(raw: str) -> Dict[str, Any]:
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

    def _build_prompt(self, sa, sb, attacks_a, attacks_b, profile):
        def fmt_atk(attacks, label):
            lines = [f"\n### 红军对 {label} 的攻击:"]
            for atk in attacks:
                lines.append(f"- [{atk.get('persona_name','?')}] 严重度={atk.get('severity','?')} | 认可度={atk.get('agreement',0):.0%}")
                for p in atk.get("attack_points",[]): lines.append(f"  • {p}")
                for f in atk.get("suggested_fixes",[]): lines.append(f"  → 建议: {f}")
            return "\n".join(lines)
        return f"## 策略辩论裁判任务\n### 策略 A: {sa.get('strategy_name','?')} ({sa.get('approach_type','?')})\n{fmt_atk(attacks_a, '策略A')}\n\n### 策略 B: {sb.get('strategy_name','?')} ({sb.get('approach_type','?')})\n{fmt_atk(attacks_b, '策略B')}\n\n### 请裁决"

    def _fallback(self, sa, sb, attacks_a, attacks_b, err):
        def avg_agr(attacks):
            vals = [a.get("agreement",0.5) for a in attacks]
            return sum(vals)/len(vals) if vals else 0.5
        def count_crit(attacks):
            return sum(1 for a in attacks if a.get("severity")=="critical")
        aa, ab = avg_agr(attacks_a)-count_crit(attacks_a)*0.15, avg_agr(attacks_b)-count_crit(attacks_b)*0.15
        sa_score, sb_score = round(aa*100), round(ab*100)
        winner = sa["strategy_name"] if sa_score>sb_score else sb["strategy_name"] if sb_score>sa_score else "tie"
        return {"strategy_a_score":sa_score,"strategy_b_score":sb_score,"winner":winner,
                "key_deciding_factor":f"启发式裁决(LLM不可用: {err[:80]})",
                "judge_commentary":f"红军认可度: A={aa:.2f}, B={ab:.2f}","suggestions_for_loser":["请人工审查"]}

    @staticmethod
    def _validate_winner(verdict, name_a, name_b):
        """校验LLM返回的winner是否为实际策略名。"""
        winner = str(verdict.get("winner", "")).strip()
        if winner == name_a or winner == name_b:
            return verdict
        if winner.lower() in ("strategy_a", "strategy a", "a"):
            verdict["winner"] = name_a; return verdict
        if winner.lower() in ("strategy_b", "strategy b", "b"):
            verdict["winner"] = name_b; return verdict
        if name_a in winner or (len(winner) < 3 and winner.upper() == "A"):
            verdict["winner"] = name_a; return verdict
        if name_b in winner or (len(winner) < 3 and winner.upper() == "B"):
            verdict["winner"] = name_b; return verdict
        sa, sb = verdict.get("strategy_a_score", 50), verdict.get("strategy_b_score", 50)
        verdict["winner"] = name_a if sa > sb else name_b if sb > sa else "tie"
        return verdict

    @staticmethod
    def update_elo(elo_ratings, winner_name, loser_name, k_factor=32.0):
        ra = elo_ratings.get(winner_name,1500.0); rb = elo_ratings.get(loser_name,1500.0)
        ea = 1.0/(1.0+10.0**((rb-ra)/400.0)); shift = k_factor*(1.0-ea)
        return shift, shift


def judge_node(state: dict) -> dict:
    from datetime import datetime as dt
    debate_result = state.get("_current_debate_result",{})
    pair = state.get("_current_debate_pair",["",""])
    if not debate_result or not pair[0]:
        return {"event_log":["[Judge] No debate to judge."]}
    strategies = {s["strategy_name"]:s for s in state.get("candidate_strategies",[])}
    sa, sb = strategies.get(pair[0],{}), strategies.get(pair[1],{})
    elo = dict(state.get("tournament_state",{}).get("elo_ratings",{}))
    judge = StrategyJudge()
    verdict = judge.judge_debate(sa, sb, debate_result.get("attacks_on_a",[]),
                                  debate_result.get("attacks_on_b",[]), state.get("target_profile"))
    # Elo
    winner_name = verdict.get("winner","")
    if winner_name and winner_name!="tie":
        loser_name = pair[1] if winner_name==pair[0] else pair[0]
        gain, _ = StrategyJudge.update_elo(elo, winner_name, loser_name, state["tournament_state"]["elo_k_factor"])
        elo[winner_name] = elo.get(winner_name,1500.0)+gain
        elo[loser_name] = elo.get(loser_name,1500.0)-gain
    leader = max(elo.items(),key=lambda x:x[1])[0] if elo else ""
    record = {"round_id":f"round_{state['tournament_state']['completed_debates']+1}",
              "strategy_a":pair[0],"strategy_b":pair[1],
              "expert_attacks_on_a":debate_result.get("attacks_on_a",[]),
              "expert_attacks_on_b":debate_result.get("attacks_on_b",[]),
              "judge_summary":verdict.get("judge_commentary",""),"winner":winner_name,
              "elo_shift_a":elo.get(pair[0],1500.0)-state["tournament_state"]["elo_ratings"].get(pair[0],1500.0),
              "elo_shift_b":elo.get(pair[1],1500.0)-state["tournament_state"]["elo_ratings"].get(pair[1],1500.0),
              "key_deciding_factor":verdict.get("key_deciding_factor",""),
              "timestamp":dt.now(timezone.utc).isoformat()}
    now = dt.now(timezone.utc).isoformat()
    return {"pipeline_stage":"tournament","tournament_state":{**state["tournament_state"],
            "elo_ratings":elo,"completed_debates":state["tournament_state"]["completed_debates"]+1,
            "current_leader":leader,"round_number":state["tournament_state"]["round_number"]+1},
            "tournament_history":state.get("tournament_history",[])+[record],
            "_current_debate_pair":[],"_current_debate_result":{},
            "updated_at":now,"event_log":[f"[{now}] [Judge] {pair[0]} vs {pair[1]}: Winner={winner_name or 'tie'}"]}
