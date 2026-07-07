"""
AutoVS-Agent: LangGraph 工作流图定义
=====================================
8 步闭环虚拟筛选漏斗的状态机拓扑。

节点 (Nodes):
  N1  strategy_node       — Step 1: Strategy Agent 战前侦察
  N2  clustering_node     — Step 2: 化学空间聚类降维
  N3  watchdog_node       — Step 3: Watchdog 小样本演习 + 参数锁定
  N4  htvs_node           — Step 4: 高通量虚拟筛选
  N5  medchem_node        — Step 5: MedChem Committee 绝对值淘汰
  N6  ranking_node        — Step 6: MPO Elo 锦标赛排序
  N7  md_oracle_node      — Step 7: MD 模拟终极验证
  N8  meta_review_node    — Step 8: Meta-Review 闭环复盘

条件边 (Conditional Edges):
  - watchdog_node → htvs_node          (演习成功)
  - watchdog_node → watchdog_node       (演习失败, 重试纠错, 最多 N 次)
  - medchem_node → ranking_node         (存活分子充足)
  - medchem_node → strategy_node        (存活分子过少, 策略需放宽)
  - meta_review_node → clustering_node  (继续下一轮迭代)
  - meta_review_node → END              (收敛 / 达到最大轮次)

使用方式:
  from src.graph.workflow import create_workflow
  app = create_workflow()
  final_state = app.invoke(initial_state)
"""

from __future__ import annotations

from typing import Dict, Literal

from langgraph.graph import END, StateGraph

from src.graph.state import (
    MACVSState,
    is_converged,
    is_pipeline_complete,
    update_timestamp,
)


# =============================================================================
# 节点函数 (Node Functions)
# =============================================================================
# 每个节点接收当前 MACVSState，返回 dict partial update。
# 实际 LLM 调用逻辑在 src/agents/ 中实现，此处为图拓扑层调用占位符。
# =============================================================================

def strategy_node(state: MACVSState) -> Dict:
    """Step 1: Strategy Agent — 战前侦察。

    委托给 src.agents.target_scout.scouting_node 执行完整的
    TargetInfo 提取 → LLM 结构化推理 → Rulebook 产出流程。

    返回的 partial state update 包含:
      - filter_protocol: Pydantic-validated DynamicFilterProtocol dict
      - protocol_version: 自增版本号
      - pipeline_stage: "strategy" (或 "error")
    """
    from src.agents.target_scout import scouting_node

    return scouting_node(state)


def clustering_node(state: MACVSState) -> Dict:
    """Step 2: 化学空间聚类降维。

    对百亿大库进行 Butina/K-Means 聚类，抽取 ~10 万个化学空间差异最大的
    代表分子进入候选池。如有闭环知识库指导，则定向偏置采样。
    """
    from src.agents.orchestrator import OrchestratorAgent

    agent = OrchestratorAgent()
    candidates = agent.run_clustering(
        library_path=state["full_library_path"],
        method=state.get("cluster_method", "Butina"),
        target_size=100000,
        knowledge_base=state.get("knowledge_base"),
    )

    return {
        "pipeline_stage": "clustering",
        "candidate_pool": candidates,
        "surviving_pool": candidates,
        "cluster_count": state.get("cluster_count", 0),
        "event_log": [f"[Clustering] {len(candidates)} representatives selected."],
        **update_timestamp(state),
    }


def watchdog_node(state: MACVSState) -> Dict:
    """Step 3: Watchdog — 小样本演习 + 参数锁定。

    选取 <10 个已知分子（阳性对照 + 诱饵 decoy）进行对接和 MD 试跑，
    自动纠错并锁定最优 Grid Box、对接穷举度、MD 力场等参数。

    如果演习失败，自动调整参数重试 (self-correction loop)。
    重试次数超限则标记异常并跳转至 error。
    """
    from src.agents.watchdog import WatchdogAgent

    agent = WatchdogAgent()
    result = agent.run_dry_run(
        target_info=state["target_info"],
        positive_control_smiles=None,  # 由 agent 从知识库或用户输入获取
        decoy_smiles_list=None,
        max_retries=state["watchdog_max_retries"],
        retry_count=state["watchdog_retry_count"],
    )

    if result.get("success"):
        return {
            "pipeline_stage": "watchdog",
            "watchdog_config": result.get("config"),
            "watchdog_retry_count": 0,
            "event_log": [f"[Watchdog] Dry-run passed. Params locked."],
            **update_timestamp(state),
        }
    else:
        new_retry = state["watchdog_retry_count"] + 1
        if new_retry >= state["watchdog_max_retries"]:
            return {
                "pipeline_stage": "error",
                "errors": [{
                    "node": "watchdog",
                    "timestamp": state["updated_at"],
                    "message": f"Watchdog dry-run failed after {new_retry} retries.",
                }],
                "event_log": [f"[Watchdog] MAX RETRIES EXCEEDED. Pipeline halted."],
                **update_timestamp(state),
            }
        return {
            "watchdog_retry_count": new_retry,
            "event_log": [f"[Watchdog] Retry {new_retry}/{state['watchdog_max_retries']}..."],
            **update_timestamp(state),
        }


def htvs_node(state: MACVSState) -> Dict:
    """Step 4: 高通量虚拟筛选 (HTVS)。

    基于 Watchdog 锁定的参数，对候选池 (~10万分子) 进行 GNINA/smina 粗对接。
    保留 Top 2000 进入下一轮。
    """
    from src.agents.watchdog import WatchdogAgent

    agent = WatchdogAgent()
    result = agent.run_htvs(
        molecules=state["candidate_pool"],
        watchdog_config=state["watchdog_config"],
        top_n=state["htvs_top_n"],
    )

    survivors = result.get("survivors", [])
    return {
        "pipeline_stage": "htvs",
        "surviving_pool": survivors,
        "htvs_total_docked": len(state["candidate_pool"]),
        "htvs_completed": True,
        "event_log": [f"[HTVS] {len(survivors)}/{len(state['candidate_pool'])} passed coarse docking."],
        **update_timestamp(state),
    }


def medchem_node(state: MACVSState) -> Dict:
    """Step 5: MedChem Committee — 三级联漏斗绝对值淘汰。

    委托给 src.agents.expert_committee.medchem_filter_node 执行:
      Tier 0: 动态对接阈值截断 (阳性对照导向)
      Tier 1: 2D 物理化学一票否决 (RDKit + SMARTS 黑名单)
      Tier 2: 3D 与 AI 多维深度图谱 (PLIP/ML/ADMET Mock)

    返回的 partial state update 包含:
      - surviving_pool: 连过三关的分子
      - committee_reports: 各 Tier 统计报告
      - screened_records: 归档被淘汰分子
    """
    from src.agents.expert_committee import medchem_filter_node

    return medchem_filter_node(state)


def ranking_node(state: MACVSState) -> Dict:
    """Step 6: MPO Elo 锦标赛 — 相对值排序。

    Ranking Agent 将存活的 ~300 个分子进行多轮 1v1 科学辩论，
    基于"亲和力、成药性、新颖性"三维雷达图更新 Elo 积分。
    保留 Top 20 进入 MD Oracle。
    """
    from src.agents.judge_agent import RankingAgent

    agent = RankingAgent()
    result = agent.run_tournament(
        molecules=state["surviving_pool"],
        dimensions=state["mpo_dimensions"],
        rounds=state["al_state"]["tournament_rounds"],
        k_factor=state["al_state"]["elo_k_factor"],
        initial_rating=state["al_state"]["elo_initial_rating"],
    )

    top_molecules = result.get("top_n", [])
    return {
        "pipeline_stage": "ranking",
        "surviving_pool": top_molecules,
        "elo_leaderboard": result.get("leaderboard", {}),
        "tournament_bracket": result.get("bracket", []),
        "event_log": [
            f"[MPO] Tournament complete. Top Elo: "
            f"{max(result.get('leaderboard', {}).values()) if result.get('leaderboard') else 'N/A'}"
        ],
        **update_timestamp(state),
    }


def md_oracle_node(state: MACVSState) -> Dict:
    """Step 7: MD Oracle — 终极神谕验证。

    对 Top 20 分子进行 50ns 全原子 MD 模拟，计算:
      - MM/GBSA 结合自由能 (ΔG)
      - 关键氢键占有率
      - 配体 RMSD 稳定性
    确定 3-5 个最终 Hit。
    """
    from src.agents.watchdog import WatchdogAgent

    agent = WatchdogAgent()
    result = agent.run_md_simulations(
        molecules=state["surviving_pool"],
        target_info=state["target_info"],
        watchdog_config=state["watchdog_config"],
        simulation_time_ns=state["md_min_simulation_ns"],
    )

    md_passed = result.get("passed", [])
    md_results = result.get("results", {})

    # 更新通过 MD 的分子标记
    for mol in state["surviving_pool"]:
        if mol["mol_id"] in md_results:
            mol["md_passed"] = mol["mol_id"] in {m["mol_id"] for m in md_passed}

    best_dG = min(
        (r.get("dG_mmgbsa", 0.0) for r in md_results.values()),
        default=0.0,
    )

    return {
        "pipeline_stage": "md_oracle",
        "md_results": md_results,
        "surviving_pool": md_passed,
        "md_passed_count": len(md_passed),
        "event_log": [
            f"[MD Oracle] {len(md_passed)}/{len(state['surviving_pool'])} passed MD. "
            f"Best ΔG: {best_dG:.2f} kcal/mol"
        ],
        **update_timestamp(state),
    }


def meta_review_node(state: MACVSState) -> Dict:
    """Step 8: Meta-Review — 闭环复盘。

    提取 MD 验证 Hit 的优势特征和动力学教训，
    更新全局知识库，决定是否启动下一轮迭代。
    如继续: 跳转回 Step 2 (定向聚类挖掘)
    如停止: 跳转到 END
    """
    from src.agents.meta_review import MetaReviewAgent

    agent = MetaReviewAgent()
    result = agent.review_and_decide(
        md_passed_hits=state["surviving_pool"],
        md_results=state["md_results"],
        knowledge_base=state["knowledge_base"],
        al_state=state["al_state"],
        tournament_bracket=state["tournament_bracket"],
    )

    # 更新闭环知识库
    updated_kb = result.get("knowledge_base", state["knowledge_base"])

    # 更新主动学习状态
    al = state["al_state"]
    new_iteration = al["iteration"] + 1
    best_dG_history = list(al["best_dG_history"]) + [result.get("best_dG", 0.0)]
    hit_rate = (
        len(state["surviving_pool"]) / max(len(state["surviving_pool"]), 1)
    )
    hit_rate_history = list(al["hit_rate_history"]) + [hit_rate]

    should_continue = result.get("continue", False) and not is_converged({
        **state,
        "al_state": {
            **al,
            "iteration": new_iteration,
            "best_dG_history": best_dG_history,
        },
    })

    if should_continue:
        return {
            "pipeline_stage": "clustering",  # 闭环回 Step 2
            "knowledge_base": updated_kb,
            "continue_next_iteration": True,
            "al_state": {
                **al,
                "iteration": new_iteration,
                "best_dG_history": best_dG_history,
                "hit_rate_history": hit_rate_history,
            },
            "final_hits": state["surviving_pool"],
            "event_log": [
                f"[Meta-Review] Iteration {new_iteration}: continuing to next round. "
                f"Best ΔG trend: {best_dG_history}"
            ],
            **update_timestamp(state),
        }
    else:
        return {
            "pipeline_stage": "converged",
            "knowledge_base": updated_kb,
            "continue_next_iteration": False,
            "al_state": {
                **al,
                "iteration": new_iteration,
                "best_dG_history": best_dG_history,
                "hit_rate_history": hit_rate_history,
            },
            "final_hits": state["surviving_pool"],
            "output_report_path": result.get("report_path"),
            "event_log": [
                f"[Meta-Review] Converged after {new_iteration} iterations. "
                f"Final hits: {len(state['surviving_pool'])}"
            ],
            **update_timestamp(state),
        }


def error_node(state: MACVSState) -> Dict:
    """异常处理节点 —— 记录错误并终止管道。"""
    return {
        "pipeline_stage": "error",
        "event_log": ["[Pipeline] Terminated due to unrecoverable error."],
        **update_timestamp(state),
    }


# =============================================================================
# 条件路由函数 (Router Functions)
# =============================================================================

def watchdog_router(state: MACVSState) -> Literal["htvs", "watchdog", "error"]:
    """Watchdog 演习后的路由决策。

    Returns:
        "htvs"     — 演习成功，进入 HTVS 阶段
        "watchdog" — 演习失败，重试纠错
        "error"    — 超过最大重试次数，终止管道
    """
    if state["pipeline_stage"] == "error":
        return "error"
    if state["watchdog_config"] is not None:
        return "htvs"
    return "watchdog"


def medchem_router(state: MACVSState) -> Literal["ranking", "strategy", "error"]:
    """MedChem 过滤后的路由决策。

    Returns:
        "ranking"  — 存活分子充足，进入锦标赛
        "strategy" — 存活分子过少，回退到 Step 1 放宽策略
        "error"    — 异常
    """
    if state["pipeline_stage"] == "error":
        return "error"
    if state["pipeline_stage"] == "strategy":
        # medchem_node 已经设置了 strategy 回退
        return "strategy"
    if len(state["surviving_pool"]) >= 10:
        return "ranking"
    return "strategy"


def meta_review_router(state: MACVSState) -> Literal["clustering", "error", "end"]:
    """Meta-Review 后的路由决策。

    Returns:
        "clustering" — 继续下一轮迭代 (回 Step 2)
        "error"      — 异常
        "end"        — 收敛 / 达到最大轮次，结束管道
    """
    if state["pipeline_stage"] == "error":
        return "error"
    if state["pipeline_stage"] == "converged":
        return "end"
    if state.get("continue_next_iteration"):
        return "clustering"
    return "end"


# =============================================================================
# 图构建: 装配 8 步漏斗
# =============================================================================

def create_workflow() -> StateGraph:
    """创建并编译 AutoVS-Agent 8 步漏斗的 LangGraph 工作流。

    Returns:
        编译后的 StateGraph 实例 (Runnable)，可直接 .invoke(initial_state)。

    Graph 拓扑:
        strategy → clustering → watchdog → [router] → htvs → medchem
        → [router] → ranking → md_oracle → meta_review → [router]
        → clustering (循环) / END

        watchdog [router]:
          ├─ 成功 → htvs
          ├─ 失败 → watchdog (重试)
          └─ 超限 → error → END

        medchem [router]:
          ├─ 充足 → ranking
          └─ 不足 → strategy (放宽规则回退)

        meta_review [router]:
          ├─ 继续 → clustering (下一轮迭代)
          └─ 收敛 → END
    """
    # ---- 初始化状态图 ----
    workflow = StateGraph(MACVSState)

    # ---- 注册节点 ----
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("clustering", clustering_node)
    workflow.add_node("watchdog", watchdog_node)
    workflow.add_node("htvs", htvs_node)
    workflow.add_node("medchem_filter", medchem_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("md_oracle", md_oracle_node)
    workflow.add_node("meta_review", meta_review_node)
    workflow.add_node("error_handler", error_node)

    # ---- 入口点 ----
    workflow.set_entry_point("strategy")

    # ---- 固定边 (确定性流转) ----
    workflow.add_edge("strategy", "clustering")
    workflow.add_edge("clustering", "watchdog")
    workflow.add_edge("htvs", "medchem_filter")
    workflow.add_edge("ranking", "md_oracle")
    workflow.add_edge("md_oracle", "meta_review")

    # ---- 条件边 (带路由的流转) ----

    # Watchdog → HTVS (成功) / Watchdog (重试) / error (超限)
    workflow.add_conditional_edges(
        "watchdog",
        watchdog_router,
        {
            "htvs": "htvs",
            "watchdog": "watchdog",
            "error": "error_handler",
        },
    )

    # MedChem → Ranking (存活充足) / Strategy (放宽回退) / error
    workflow.add_conditional_edges(
        "medchem_filter",
        medchem_router,
        {
            "ranking": "ranking",
            "strategy": "strategy",
            "error": "error_handler",
        },
    )

    # Meta-Review → Clustering (下一轮) / END (收敛)
    workflow.add_conditional_edges(
        "meta_review",
        meta_review_router,
        {
            "clustering": "clustering",
            "error": "error_handler",
            "end": END,
        },
    )

    # Error → END
    workflow.add_edge("error_handler", END)

    # ---- 编译 ----
    app = workflow.compile()
    return app


# =============================================================================
# 便捷入口
# =============================================================================

def run_pipeline(initial_state: MACVSState) -> MACVSState:
    """一键运行 8 步漏斗管道。

    Args:
        initial_state: 由 create_initial_state() 创建的初始状态。

    Returns:
        管道完成后的最终 MACVSState。
    """
    app = create_workflow()
    final_state = app.invoke(initial_state)
    return final_state
