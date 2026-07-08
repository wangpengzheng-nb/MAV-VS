"""
src.agents — AutoVS-Agent v2.0 多智能体协作层
================================================
v2 新增:
  - TargetScoutAgent: 靶点深度侦察
  - StrategyGeneratorAgent: 多策略生成
  - RedTeamReviewer: 红军三人设辩论评审
  - StrategyJudge: 策略级 Elo 锦标赛裁判
"""

from src.agents.target_scout import TargetScoutAgent
from src.agents.strategy_generator import StrategyGeneratorAgent
from src.agents.expert_committee import RedTeamReviewer
from src.agents.judge_agent import StrategyJudge
from src.agents.orchestrator import OrchestratorAgent
from src.agents.watchdog import WatchdogAgent
from src.agents.meta_review import MetaReviewAgent
from src.agents.proxy_mlp import ProxyMLP

__all__ = [
    "TargetScoutAgent",
    "StrategyGeneratorAgent",
    "RedTeamReviewer",
    "StrategyJudge",
    "OrchestratorAgent",
    "WatchdogAgent",
    "MetaReviewAgent",
    "ProxyMLP",
]
