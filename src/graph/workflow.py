"""
AutoVS-Agent v2.0: 锦标赛工作流图
==================================
拓扑: TargetScout → StrategyGeneration → [Tournament Loop] → MetaReview

Tournament Loop:
  red_team_debate (红军评审) → judge (裁判裁决)
  → [router: more_pairings?  → red_team_debate
           : done            → meta_review]
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Literal

from langgraph.graph import END, StateGraph

from src.graph.state import MACVSState, update_timestamp


# =============================================================================
# 节点函数
# =============================================================================

def target_scout_node(state: MACVSState) -> Dict:
    """Step 1: Target Scout — 靶点深度侦察。"""
    from src.agents.target_scout import target_scout_node as _node
    return _node(state)


def strategy_generation_node(state: MACVSState) -> Dict:
    """Step 2: Strategy Generation — 多策略生成。"""
    from src.agents.strategy_generator import strategy_generation_node as _node
    return _node(state)


def red_team_debate_node(state: MACVSState) -> Dict:
    """Step 3a: Red Team 红军评审 — 三人设同时攻击两个策略。"""
    from src.agents.expert_committee import red_team_debate_node as _node
    return _node(state)


def judge_node(state: MACVSState) -> Dict:
    """Step 3b: Judge 裁判 — 裁决辩论 + 更新 Elo。"""
    from src.agents.judge_agent import judge_node as _node
    return _node(state)


def meta_review_node(state: MACVSState) -> Dict:
    """Step 4: Meta Review — 最佳策略进化。"""
    from src.agents.meta_review import MetaReviewAgent

    strategies = {s["strategy_name"]: s for s in state.get("candidate_strategies", [])}
    elo = state.get("tournament_state", {}).get("elo_ratings", {})
    history = state.get("tournament_history", [])

    if not strategies:
        return {"pipeline_stage": "converged", "event_log": ["[MetaReview] No strategies to review."]}

    # 选 Elo 最高策略
    leader_name = max(elo.items(), key=lambda x: x[1])[0] if elo else list(strategies.keys())[0]
    best = dict(strategies.get(leader_name, {}))

    # 收集所有改进建议
    all_suggestions = []
    for debate in history:
        for atk_list in [debate.get("expert_attacks_on_a", []), debate.get("expert_attacks_on_b", [])]:
            for atk in atk_list:
                all_suggestions.extend(atk.get("suggested_fixes", []))

    # 去重
    unique_suggestions = list(dict.fromkeys(all_suggestions))[:10]

    agent = MetaReviewAgent()
    result = agent.review_and_decide(
        md_passed_hits=[], md_results={},
        knowledge_base=state.get("knowledge_base", {}),
        al_state=state.get("al_state", {}),
        tournament_bracket=state.get("tournament_bracket", []),
    )

    now = datetime.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "converged",
        "best_strategy": {
            **best,
            "elo_rating": elo.get(leader_name, 1500.0),
            "debate_count": len(history),
            "evolution_suggestions": unique_suggestions,
            "meta_review_approved": True,
        },
        "updated_at": now,
        "event_log": [
            f"[{now}] [MetaReview] Best strategy: {leader_name} "
            f"(Elo={elo.get(leader_name, 0):.0f}). "
            f"Suggestions gathered: {len(unique_suggestions)}. "
            f"Pipeline converged."
        ],
    }


# =============================================================================
# 路由函数
# =============================================================================

def tournament_router(state: MACVSState) -> Literal["red_team_debate", "meta_review"]:
    """锦标赛路由: 还有配对 → 继续辩论, 否则 → MetaReview。"""
    pairings = state.get("tournament_state", {}).get("pairings_queue", [])
    max_rounds = state.get("tournament_state", {}).get("max_rounds", 6)
    completed = state.get("tournament_state", {}).get("completed_debates", 0)

    if pairings and completed < max_rounds:
        return "red_team_debate"
    return "meta_review"


# =============================================================================
# 图构建
# =============================================================================

def create_workflow() -> StateGraph:
    """创建锦标赛版 LangGraph 工作流。

    拓扑:
      target_scout → strategy_generation → judge
        → [router: more? → red_team_debate → judge | done → meta_review → END]
    """
    workflow = StateGraph(MACVSState)

    # 注册节点
    workflow.add_node("target_scout", target_scout_node)
    workflow.add_node("strategy_generation", strategy_generation_node)
    workflow.add_node("red_team_debate", red_team_debate_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("meta_review", meta_review_node)

    # 入口
    workflow.set_entry_point("target_scout")

    # 固定边
    workflow.add_edge("target_scout", "strategy_generation")
    workflow.add_edge("strategy_generation", "judge")   # 先裁决初始状态
    workflow.add_edge("red_team_debate", "judge")

    # 条件边: judge → [循环 or MetaReview]
    workflow.add_conditional_edges(
        "judge",
        tournament_router,
        {
            "red_team_debate": "red_team_debate",
            "meta_review": "meta_review",
        },
    )

    # 终点
    workflow.add_edge("meta_review", END)

    return workflow.compile()


def run_pipeline(initial_state: MACVSState) -> MACVSState:
    """一键运行锦标赛管道。"""
    app = create_workflow()
    return app.invoke(initial_state)
