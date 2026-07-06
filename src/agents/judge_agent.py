"""
AutoVS-Agent: Ranking Agent (裁判智能体)
==========================================
职责 (Step 6: MPO 巅峰锦标赛):
  - 将存活分子进行多轮 1v1 科学辩论
  - 基于三维雷达图 (亲和力、成药性、新颖性) 进行多参数优化 (MPO)
  - 使用 Elo 评分系统更新分子积分
  - 保留 Top 20 进入 MD Oracle

输入:
  - 存活分子池 (~300)
  - 评分维度定义
  - 锦标赛配置 (轮次、K 因子)

输出:
  - Elo 积分榜 (leaderboard)
  - 锦标赛对阵记录 (bracket)
  - Top 20 分子列表
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.graph.state import (
    MoleculeRecord,
    MoleculeID,
    TournamentMatch,
)


class RankingAgent:
    """Ranking Agent — Elo 锦标赛裁判。

    核心机制:
      - 瑞士制 / 循环赛配对
      - LLM 驱动的 1v1 科学辩论
      - 三维雷达图 (亲和力/成药性/新颖性) 综合评分
      - Elo 评分系统动态更新
    """

    def __init__(
        self,
        llm_model: str = "deepseek-chat",
        llm_api_base: Optional[str] = None,
        llm_temperature: float = 0.2,
    ):
        """
        Args:
            llm_model: LLM 模型名称。
            llm_api_base: API endpoint。
            llm_temperature: 裁判任务需要一致性和公平性。
        """
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.llm_temperature = llm_temperature
        self._client = None

    # -------------------------------------------------------------------------
    # 主入口: 运行锦标赛
    # -------------------------------------------------------------------------

    def run_tournament(
        self,
        molecules: List[MoleculeRecord],
        dimensions: Optional[List[str]] = None,
        rounds: int = 3,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ) -> Dict[str, Any]:
        """Step 6 主方法: 运行 MPO Elo 锦标赛。

        流程:
          1. 初始化所有分子的 Elo 积分 (1500)
          2. 进行 N 轮配对 (瑞士制)
          3. 每轮: 两两配对 → LLM 辩论 → 更新 Elo
          4. 选取 Top-N (默认 20)

        Args:
            molecules: 存活分子池。
            dimensions: 评分维度 ["affinity", "druglikeness", "novelty"]。
            rounds: 锦标赛轮次。
            k_factor: Elo K 因子。
            initial_rating: 初始 Elo 积分。

        Returns:
            {
                "leaderboard": Dict[MoleculeID, float],
                "bracket": List[TournamentMatch],
                "top_n": List[MoleculeRecord],
            }
        """
        if dimensions is None:
            dimensions = ["affinity", "druglikeness", "novelty"]

        # 初始化 Elo
        elo: Dict[MoleculeID, float] = {
            m["mol_id"]: initial_rating for m in molecules
        }
        bracket: List[TournamentMatch] = []
        match_counter = 0

        # 锦标赛轮次
        for round_num in range(1, rounds + 1):
            # 配对: 瑞士制 (按当前 Elo 排序后相邻配对)
            pairs = self._swiss_pairing(molecules, elo)

            # 逐对辩论
            for mol_a, mol_b in pairs:
                match_counter += 1
                match = self._debate_pair(
                    mol_a=mol_a,
                    mol_b=mol_b,
                    dimensions=dimensions,
                    round_number=round_num,
                    match_id=f"match_{round_num}_{match_counter}",
                )
                bracket.append(match)

                # Elo 更新
                if match["winner"]:
                    winner_id = match["winner"]
                    loser_id = mol_b["mol_id"] if winner_id == mol_a["mol_id"] else mol_a["mol_id"]
                    shift = self._elo_shift(elo[winner_id], elo[loser_id], k_factor)
                    elo[winner_id] += shift
                    elo[loser_id] -= shift
                    match["elo_shift"] = shift

        # 按 Elo 排序取 Top-N
        sorted_by_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)
        top_n_ids = {mid for mid, _ in sorted_by_elo[:20]}
        top_n = [m for m in molecules if m["mol_id"] in top_n_ids]

        # 更新分子记录的 Elo 积分
        mol_map = {m["mol_id"]: m for m in molecules}
        for mid, rating in elo.items():
            if mid in mol_map:
                mol_map[mid]["elo_rating"] = rating

        return {
            "leaderboard": elo,
            "bracket": bracket,
            "top_n": top_n,
        }

    # -------------------------------------------------------------------------
    # 1v1 科学辩论
    # -------------------------------------------------------------------------

    def _debate_pair(
        self,
        mol_a: MoleculeRecord,
        mol_b: MoleculeRecord,
        dimensions: List[str],
        round_number: int,
        match_id: str,
    ) -> TournamentMatch:
        """让 LLM 作为裁判对两个分子进行 1v1 科学辩论。

        辩论基于三维雷达图:
          - 亲和力 (affinity): 对接分数、结构互补性
          - 成药性 (druglikeness): ADMET 参数
          - 新颖性 (novelty): 与已知药物的结构差异

        Args:
            mol_a, mol_b: 对战的分子对。
            dimensions: 评分维度。
            round_number: 当前轮次。
            match_id: 比赛 ID。

        Returns:
            TournamentMatch
        """
        # TODO: LLM 驱动的多维科学辩论
        # prompt = self._build_debate_prompt(mol_a, mol_b, dimensions)
        # verdict = self._call_llm(prompt)
        # return self._parse_verdict(verdict, mol_a, mol_b, match_id, round_number)

        # 暂用简单启发式 (后续替换为 LLM):
        score_a = (mol_a.get("docking_score") or 0) + (mol_a.get("structural_score") or 0)
        score_b = (mol_b.get("docking_score") or 0) + (mol_b.get("structural_score") or 0)

        return TournamentMatch(
            match_id=match_id,
            mol_a=mol_a["mol_id"],
            mol_b=mol_b["mol_id"],
            winner=mol_a["mol_id"] if score_a > score_b else mol_b["mol_id"] if score_b > score_a else None,
            elo_shift=0.0,  # 由外部调用 _elo_shift 更新
            debate_summary="Heuristic placeholder — to be replaced by LLM debate.",
            dimensions={
                "affinity": {mol_a["mol_id"]: score_a, mol_b["mol_id"]: score_b},
                "druglikeness": {mol_a["mol_id"]: 0.0, mol_b["mol_id"]: 0.0},
                "novelty": {mol_a["mol_id"]: 0.0, mol_b["mol_id"]: 0.0},
            },
            round_number=round_number,
        )

    # -------------------------------------------------------------------------
    # Elo 系统
    # -------------------------------------------------------------------------

    @staticmethod
    def _elo_shift(rating_a: float, rating_b: float, k: float = 32.0) -> float:
        """计算 A 胜 B 后的 Elo 积分变化量。

        Elo 公式:
          expected_a = 1 / (1 + 10^((rating_b - rating_a) / 400))
          shift = k * (1 - expected_a)

        Args:
            rating_a: 胜者当前积分。
            rating_b: 败者当前积分。
            k: K 因子。

        Returns:
            胜者积分增加量 (正数)。
        """
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        return k * (1.0 - expected_a)

    # -------------------------------------------------------------------------
    # 配对算法
    # -------------------------------------------------------------------------

    @staticmethod
    def _swiss_pairing(
        molecules: List[MoleculeRecord],
        elo: Dict[MoleculeID, float],
    ) -> List[Tuple[MoleculeRecord, MoleculeRecord]]:
        """瑞士制配对: 按 Elo 排序后相邻配对。

        Args:
            molecules: 参选分子。
            elo: 当前 Elo 积分。

        Returns:
            配对列表 [(mol_a, mol_b), ...]
        """
        sorted_mols = sorted(molecules, key=lambda m: elo.get(m["mol_id"], 1500.0), reverse=True)
        pairs = []
        for i in range(0, len(sorted_mols) - 1, 2):
            pairs.append((sorted_mols[i], sorted_mols[i + 1]))
        # 奇数个分子: 最后一个轮空
        return pairs

    # -------------------------------------------------------------------------
    # 辩论结果解析
    # -------------------------------------------------------------------------

    def _parse_verdict(
        self,
        llm_response: str,
        mol_a: MoleculeRecord,
        mol_b: MoleculeRecord,
        match_id: str,
        round_number: int,
    ) -> TournamentMatch:
        """解析 LLM 辩论裁决结果。

        TODO: 从 LLM 输出的结构化 JSON 中提取:
          - winner: mol_id
          - dimension_scores: {dim: {mol_a: x, mol_b: y}}
          - debate_summary: str
        """
        # 占位符
        return TournamentMatch(
            match_id=match_id,
            mol_a=mol_a["mol_id"],
            mol_b=mol_b["mol_id"],
            winner=None,
            elo_shift=0.0,
            debate_summary="Pending LLM integration.",
            dimensions={},
            round_number=round_number,
        )
