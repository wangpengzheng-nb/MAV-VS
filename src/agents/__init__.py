"""
src.agents — AutoVS-Agent v3.0 多智能体协作层
================================================
v3 新增:
  - TournamentReviewer: 策略排名锦标赛三人设独立评审
  - StrategyJudge: 增强裁判 (CoT推理链 + 动态Elo + 平局)
"""

from src.agents.target_scout import TargetScoutAgent
from src.agents.strategy_generator import StrategyGeneratorAgent
from src.agents.strategy_evolver import StrategyEvolver
from src.agents.tool_caller import ToolCallerAgent
from src.agents.expert_committee import TournamentReviewer, RedTeamReviewer
from src.agents.judge_agent import StrategyJudge
from src.agents.orchestrator import OrchestratorAgent
from src.agents.watchdog import WatchdogAgent
from src.agents.meta_review import MetaReviewAgent
from src.agents.proxy_mlp import ProxyMLP

__all__ = [
    "TargetScoutAgent",
    "StrategyGeneratorAgent",
    "StrategyEvolver",
    "TournamentReviewer",
    "RedTeamReviewer",
    "StrategyJudge",
    "OrchestratorAgent",
    "WatchdogAgent",
    "MetaReviewAgent",
    "ProxyMLP",
]
