"""
AutoVS-Agent v4: VoteAggregator — 统票 + 诊断报告 (纯Python, 不调用LLM)
===========================================================================
收集所有评审官投票 → 统票排名 → 生成Top3诊断报告 → 输入进化智能体
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from collections import defaultdict


class VoteAggregator:
    """统票器: 从 pairwise 投票中计算排名和诊断报告。"""

    def __init__(self):
        self.results: List[dict] = []

    def add_result(self, result: dict):
        """添加一个评审官的对比结果。"""
        self.results.append(result)

    def add_results(self, results: list):
        """批量添加。过滤异常。"""
        valid = [r for r in results if isinstance(r, dict) and r.get("overall_verdict")]
        self.results.extend(valid)
        failures = len(results) - len(valid)
        if failures:
            print(f"  ⚠️ VoteAggregator: {failures}/{len(results)} 个评审结果无效, 已过滤")

    # ═══════════════════════════════════════════════════
    # 统票
    # ═══════════════════════════════════════════════════

    def count_votes(self) -> Dict[str, float]:
        """统计每个策略的总得票数 (置信度加权)。

        Returns: {strategy_name: total_votes}
        """
        votes: Dict[str, float] = defaultdict(float)
        win_loss_draw: Dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "d": 0})

        for r in self.results:
            a = r.get("strategy_a", "?")
            b = r.get("strategy_b", "?")
            winner = r.get("overall_verdict", "tie")
            conf = r.get("verdict_confidence", "medium")
            w = {"high": 3, "medium": 2, "low": 1}.get(conf, 2) / 3  # 权重: 1.0/0.67/0.33

            if winner == "A":
                votes[a] += 1.0 * w
                win_loss_draw[a]["w"] += 1
                win_loss_draw[b]["l"] += 1
            elif winner == "B":
                votes[b] += 1.0 * w
                win_loss_draw[b]["w"] += 1
                win_loss_draw[a]["l"] += 1
            else:  # tie
                votes[a] += 0.5 * w
                votes[b] += 0.5 * w
                win_loss_draw[a]["d"] += 1
                win_loss_draw[b]["d"] += 1

        # 附加胜负数到 votes 中
        result = {}
        for name, v in votes.items():
            result[name] = round(v, 2)
        return result

    # ═══════════════════════════════════════════════════
    # 排名
    # ═══════════════════════════════════════════════════

    def rank(self, strategies: list) -> List[Dict[str, Any]]:
        """按得票数排名, 平票时用决胜局规则。"""
        vote_counts = self.count_votes()
        names = [s.get("strategy_name", s.get("strategy_id", "?")) for s in strategies]

        # 收集每个策略的统计
        stats = {}
        for name in names:
            concerns = self._get_concerns_for(name)
            suggestions = self._get_suggestions_for(name)
            confidences = self._get_confidences_for(name)
            fatal_count = sum(1 for c in concerns if c.get("severity") == "Fatal")
            fw_count = sum(1 for c in concerns if c.get("severity") in ("Fatal", "Warning"))
            high_suggs = sum(1 for s in suggestions if s.get("priority") == "High")
            avg_conf = sum({"high": 3, "medium": 2, "low": 1}.get(c, 2) for c in confidences) / max(len(confidences), 1)

            stats[name] = {
                "votes": vote_counts.get(name, 0),
                "fatal_count": fatal_count,
                "fw_total": fw_count,
                "high_suggestions": high_suggs,
                "avg_confidence": avg_conf,
            }

        # 排序
        def sort_key(name):
            s = stats.get(name, {})
            return (s.get("votes", 0), -s.get("fatal_count", 0),
                    -s.get("fw_total", 0), -s.get("high_suggestions", 0),
                    s.get("avg_confidence", 0))

        ranked_names = sorted(names, key=sort_key, reverse=True)
        ranking = []
        for i, name in enumerate(ranked_names):
            s = stats[name]
            wld = self._get_wld(name)
            ranking.append({
                "rank": i + 1,
                "strategy_name": name,
                "total_votes": s["votes"],
                "wins": wld["w"],
                "losses": wld["l"],
                "draws": wld["d"],
                "fatal_count": s["fatal_count"],
                "high_suggestions": s["high_suggestions"],
            })
        return ranking

    def _get_wld(self, name: str) -> dict:
        w, l, d = 0, 0, 0
        for r in self.results:
            a, b = r.get("strategy_a", ""), r.get("strategy_b", "")
            winner = r.get("overall_verdict", "tie")
            if name in (a, b):
                if winner == "tie":
                    d += 1
                elif (winner == "A" and name == a) or (winner == "B" and name == b):
                    w += 1
                else:
                    l += 1
        return {"w": w, "l": l, "d": d}

    # ═══════════════════════════════════════════════════
    # 诊断报告
    # ═══════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════
    # 进化智能体输入准备
    # ═══════════════════════════════════════════════════

    def prepare_evolution_input(self, strategy: dict, strategy_name: str) -> dict:
        """为进化智能体准备结构化输入: 策略底稿 + UUID + 聚合诊断。

        Returns:
            {"blueprint": strategy_with_uuids, "diagnosis": {...}}
        """
        import uuid as _uuid

        # 1. 确保UUID
        blueprint = dict(strategy)
        if not blueprint.get("strategy_id"):
            blueprint["strategy_id"] = f"s-{_uuid.uuid4().hex[:8]}"
        for st in blueprint.get("pipeline", []):
            if not st.get("step_id"):
                st["step_id"] = f"a-{_uuid.uuid4().hex[:8]}"

        # 2. 提取该策略的聚合诊断
        concerns = self._dedup_concerns(self._get_concerns_for(strategy_name))
        suggestions = self._dedup_suggestions(self._get_suggestions_for(strategy_name))

        # 3. 维度强弱统计
        dim_strengths = defaultdict(int)
        dim_weaknesses = defaultdict(int)
        for r in self.results:
            for dv in r.get("dimension_votes", []):
                dim = dv.get("dimension", "?")
                w = dv.get("winner", "tie")
                if w == "A" and r.get("strategy_a") == strategy_name:
                    dim_strengths[dim] += 1
                elif w == "B" and r.get("strategy_b") == strategy_name:
                    dim_strengths[dim] += 1
                elif w != "tie" and w != "N/A":
                    dim_weaknesses[dim] += 1

        return {
            "blueprint": blueprint,
            "diagnosis": {
                "strategy_name": strategy_name,
                "concerns": concerns,
                "suggestions": suggestions,
                "strengths": [f"{k}(+{v}场)" for k, v in sorted(dim_strengths.items(), key=lambda x: -x[1]) if v > 0],
                "weaknesses": [f"{k}(-{v}场)" for k, v in sorted(dim_weaknesses.items(), key=lambda x: -x[1]) if v > 0],
            }
        }

    def generate_diagnostic(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """为Top N策略生成聚合诊断报告。"""
        ranking = self.rank([]) if not hasattr(self, '_cached_ranking') else getattr(self, '_cached_ranking', [])
        top_names = [r["strategy_name"] for r in ranking[:top_n]]

        diagnostics = []
        for name in top_names:
            concerns = self._get_concerns_for(name)
            suggestions = self._get_suggestions_for(name)

            # 去重合并
            deduped_concerns = self._dedup_concerns(concerns)
            deduped_suggs = self._dedup_suggestions(suggestions)

            # 维度强弱统计
            dim_strengths = defaultdict(int)
            dim_weaknesses = defaultdict(int)
            for r in self.results:
                for dv in r.get("dimension_votes", []):
                    dim = dv.get("dimension", "?")
                    w = dv.get("winner", "tie")
                    if w == "A" and r.get("strategy_a") == name:
                        dim_strengths[dim] += 1
                    elif w == "B" and r.get("strategy_b") == name:
                        dim_strengths[dim] += 1
                    elif w != "tie" and w != "N/A":
                        dim_weaknesses[dim] += 1

            diagnostics.append({
                "strategy_name": name,
                "aggregated_concerns": deduped_concerns,
                "aggregated_suggestions": deduped_suggs,
                "dimension_strengths": [f"{k}(+{v}场)" for k, v in sorted(dim_strengths.items(), key=lambda x: -x[1]) if v > 0],
                "dimension_weaknesses": [f"{k}(-{v}场)" for k, v in sorted(dim_weaknesses.items(), key=lambda x: -x[1]) if v > 0],
            })
        return diagnostics

    # ═══════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════

    def _get_concerns_for(self, name: str) -> list:
        concerns = []
        for r in self.results:
            for side in ("A", "B"):
                if r.get(f"strategy_{side.lower()}") == name:
                    items = r.get("critical_concerns", {}).get(side, [])
                    if isinstance(items, str): items = [items]
                    for c in (items or []):
                        if isinstance(c, dict): concerns.append(c)
        return concerns

    def _get_suggestions_for(self, name: str) -> list:
        suggestions = []
        for r in self.results:
            for side in ("A", "B"):
                if r.get(f"strategy_{side.lower()}") == name:
                    items = r.get("suggestions", {}).get(side, [])
                    if isinstance(items, str): items = [items]
                    for s in (items or []):
                        if isinstance(s, dict): suggestions.append(s)
        return suggestions

    def _get_confidences_for(self, name: str) -> list:
        confs = []
        for r in self.results:
            for side in ("A", "B"):
                if r.get(f"strategy_{side.lower()}") == name:
                    confs.append(r.get("verdict_confidence", "medium"))
        return confs

    @staticmethod
    def _dedup_concerns(concerns: list) -> list:
        seen, deduped = set(), []
        for c in concerns:
            fp = f"{c.get('step_id','')}:{c.get('issue','')[:60]}:{c.get('severity','')}"
            if fp not in seen:
                seen.add(fp)
                deduped.append(c)
        deduped.sort(key=lambda x: {"Fatal": 0, "Warning": 1, "Info": 2}.get(x.get("severity", ""), 3))
        return deduped

    @staticmethod
    def _dedup_suggestions(suggestions: list) -> list:
        seen, deduped = set(), []
        for s in suggestions:
            action = s.get("action", "")
            parts = action.split()
            cmd = parts[0] if parts else ""
            sid = s.get("step_id", parts[1] if len(parts) > 1 else "?")
            fp = f"{sid}:{cmd}"
            if fp not in seen:
                seen.add(fp)
                deduped.append(s)
            else:
                # 合并 rationale
                existing = next(d for d in deduped if d.get("step_id") == sid)
                existing["rationale"] = (existing.get("rationale", "") + "; " + s.get("rationale", ""))[:300]
        deduped.sort(key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x.get("priority", ""), 3))
        return deduped


# ═══════════════════════════════════════════════════
# 旧版兼容
# ═══════════════════════════════════════════════════

class StrategyJudge:
    """旧版兼容接口 — 请使用 VoteAggregator。"""

    def __init__(self, model="deepseek-chat", **kwargs):
        pass

    def judge_match(self, *args, **kwargs):
        return {"winner": "tie", "confidence": 0.5,
                "strategy_a_score": 50, "strategy_b_score": 50}

    @staticmethod
    def compute_initial_elo(review_result: dict, base: int = 1000) -> float:
        return 1500.0

    @staticmethod
    def update_elo(elo, winner, loser, is_tie=False):
        return 0, 0, elo

    def judge_debate(self, *args, **kwargs):
        return {"winner": "tie", "confidence": 0.5}


def judge_node(state: dict) -> dict:
    return {"event_log": ["[Judge] v4 — use VoteAggregator"]}
