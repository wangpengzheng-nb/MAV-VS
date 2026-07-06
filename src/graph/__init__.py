"""
src.graph — LangGraph 状态管理与工作流拓扑
============================================
提供 AutoVS-Agent 8 步漏斗的全局状态定义和 LangGraph 工作流图。
"""

from src.graph.state import (
    # 主状态
    MACVSState,
    # 数据模型
    MoleculeRecord,
    MoleculeID,
    SMILES,
    TargetInfo,
    DynamicFilterProtocol,
    WatchdogConfig,
    MDSimulationRecord,
    TournamentMatch,
    ClosedLoopKnowledge,
    ActiveLearningState,
    ExpertMember,
    ExpertCommitteeReport,
    # 工厂函数
    create_initial_state,
    # 工具函数
    is_pipeline_complete,
    is_converged,
    get_survivor_count,
    get_top_elo,
    get_md_passed_hits,
    update_timestamp,
    export_checkpoint,
)

from src.graph.workflow import (
    create_workflow,
    run_pipeline,
)

__all__ = [
    "MACVSState",
    "MoleculeRecord",
    "MoleculeID",
    "SMILES",
    "TargetInfo",
    "DynamicFilterProtocol",
    "WatchdogConfig",
    "MDSimulationRecord",
    "TournamentMatch",
    "ClosedLoopKnowledge",
    "ActiveLearningState",
    "ExpertMember",
    "ExpertCommitteeReport",
    "create_initial_state",
    "is_pipeline_complete",
    "is_converged",
    "get_survivor_count",
    "get_top_elo",
    "get_md_passed_hits",
    "update_timestamp",
    "export_checkpoint",
    "create_workflow",
    "run_pipeline",
]
