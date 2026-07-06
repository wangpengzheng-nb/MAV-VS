"""
src.agents — AutoVS-Agent 多智能体协作层
==========================================
6 个核心智能体 + 1 个代理模型，各司其职协同完成 8 步虚拟筛选漏斗。
"""

from src.agents.orchestrator import OrchestratorAgent
from src.agents.target_scout import StrategyAgent
from src.agents.expert_committee import (
    MedChemCommittee,
    StructuralBiologist,
    ADMETSpecialist,
    MedChemSynthesis,
)
from src.agents.judge_agent import RankingAgent
from src.agents.watchdog import WatchdogAgent
from src.agents.meta_review import MetaReviewAgent
from src.agents.proxy_mlp import ProxyMLP

__all__ = [
    "OrchestratorAgent",
    "StrategyAgent",
    "MedChemCommittee",
    "StructuralBiologist",
    "ADMETSpecialist",
    "MedChemSynthesis",
    "RankingAgent",
    "WatchdogAgent",
    "MetaReviewAgent",
    "ProxyMLP",
]
