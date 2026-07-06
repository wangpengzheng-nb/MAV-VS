"""
AutoVS-Agent: Ranking Agent (裁判智能体)
==========================================
职责 (Step 6: MPO 巅峰锦标赛):
  - 将存活分子进行多轮 1v1 科学辩论
  - 基于三维雷达图 (亲和力、成药性、新颖性) 进行多参数优化 (MPO)
  - 使用 Elo 评分系统更新分子积分
  - 保留 Top 20 进入 MD Oracle
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from src.graph.state import (
    MoleculeRecord,
    MoleculeID,
    TournamentMatch,
)

# =============================================================================
# 1. 结构化输出模型 (Pydantic Schema)
# =============================================================================

class RadarScore(BaseModel):
    affinity_score: int = Field(..., description="结合亲和力得分 (1-10分)，基于对接分数和关键氢键/药效团匹配度。")
    admet_score: int = Field(..., description="成药性与稳定性得分 (1-10分)，基于MW, LogP及PAINS警示。存在严重毒性预警则必须打低分。")
    novelty_score: int = Field(..., description="骨架新颖性得分 (1-10分)，基于与已知药物的结构差异。")

class DebateVerdict(BaseModel):
    molecule_A_scores: RadarScore = Field(..., description="分子A的三维雷达图得分")
    molecule_B_scores: RadarScore = Field(..., description="分子B的三维雷达图得分")
    rationale: str = Field(..., description="综合评判理由，必须明确指出胜负的关键分歧点（如：A亲和力虽好但毒性高，故B胜出）。字数在150字以内。")
    winner_id: str = Field(..., description="最终获胜分子的 ID (必须是 molecule_A 或 molecule_B 的 ID)")

DebateVerdict.model_rebuild()

# =============================================================================
# 2. 核心裁判提示词 (System Prompt)
# =============================================================================

JUDGE_SYSTEM_PROMPT = """\
你是一位顶尖的计算化学家和医药公司首席科学家 (CSO)。
目前你正在主持一场靶向药物发现的“多参数优化 (MPO) 锦标赛”。你的任务是作为核心裁判，对两两对决的候选小分子（分子A 和 分子B）进行 1v1 的评估，并决出胜者。

请严格基于以下三个维度的权重逻辑进行裁决，并为每个分子在各维度打分（1-10分）：

1. 【成药性与底线审查 (ADMET & Stability)】- 惩罚性维度 (最重要)
- 裁判准则：如果某分子存在明显的代谢毒性预警、极端的LogP/MW，或违反药化红线，无论其亲和力多高，必须在此项打极低分，并判负。

2. 【结合亲和力 (Binding Affinity)】- 核心动力
- 裁判准则：比较对接总分的绝对值 (越负越好)，以及结构互补性。分数更优者得分更高。

3. 【骨架新颖性 (Scaffold Novelty)】- 溢价加分项
- 裁判准则：如果两个分子在亲和力和成药性上势均力敌，提供全新化学骨架的分子应获得显著加分并胜出。

【严格约束】：
1. 绝对客观：严禁脱离提供的数据凭空捏造分子的性质。
2. 强制分出胜负：不允许平局，必须基于 MPO 原则选出一个综合价值最高的分子。
3. 结构化输出：必须严格按照要求的 JSON 格式输出，`winner_id` 必须准确无误。
"""

# =============================================================================
# 3. Ranking Agent 类
# =============================================================================

class RankingAgent:
    """Ranking Agent — Elo 锦标赛裁判。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,  # 降低温度，保证裁判的稳定性
        max_tokens: int = 1024,
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
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=f"{self.api_base}/v1",
            )
        return self._client

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
        if dimensions is None:
            dimensions = ["affinity", "druglikeness", "novelty"]

        # 初始化 Elo
        elo: Dict[MoleculeID, float] = {
            m["mol_id"]: m.get("elo_rating") or initial_rating for m in molecules
        }
        bracket: List[TournamentMatch] = []
        match_counter = 0

        # 锦标赛轮次
        for round_num in range(1, rounds + 1):
            pairs = self._swiss_pairing(molecules, elo)

            # 逐对辩论
            for mol_a, mol_b in pairs:
                match_counter += 1
                match = self._debate_pair(
                    mol_a=mol_a,
                    mol_b=mol_b,
                    round_number=round_num,
                    match_id=f"match_{round_num}_{match_counter}",
                )
                bracket.append(match)

                # Elo 更新 (Python计算，不依赖大模型)
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

        # 更新分子记录的 Elo 积分和胜负场次
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
    # 1v1 科学辩论 (LLM 调用)
    # -------------------------------------------------------------------------
    def _debate_pair(
        self,
        mol_a: MoleculeRecord,
        mol_b: MoleculeRecord,
        round_number: int,
        match_id: str,
    ) -> TournamentMatch:
        """调用 LLM 进行 1v1 科学辩论，并强制输出结构化结果。"""
        
        # 提取双方的客观数据构建 User Prompt
        def extract_features(mol: MoleculeRecord) -> str:
            return (f"ID: {mol['mol_id']} | "
                    f"Docking Score: {mol.get('docking_score', 'N/A')} | "
                    f"ADMET Flags: {mol.get('admet_flags', 'None')}")

        user_prompt = f"""
        请对以下两个分子进行 MPO 综合评估并决出胜者：
        【分子 A】: {extract_features(mol_a)}
        【分子 B】: {extract_features(mol_b)}
        """

        try:
            # 强制 JSON 结构化输出
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "system",
                        "content": f"请严格输出 JSON，Schema如下：\n{json.dumps(DebateVerdict.model_json_schema())}"
                    }
                ],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            verdict = DebateVerdict.model_validate(parsed)
            
            # 安全校验：防止 LLM 幻觉编造了一个不存在的 winner_id
            winner_id = verdict.winner_id
            if winner_id not in [mol_a["mol_id"], mol_b["mol_id"]]:
                winner_id = mol_a["mol_id"] # 兜底逻辑

            return TournamentMatch(
                match_id=match_id,
                mol_a=mol_a["mol_id"],
                mol_b=mol_b["mol_id"],
                winner=winner_id,
                elo_shift=0.0, 
                debate_summary=verdict.rationale,
                dimensions={
                    "affinity": {mol_a["mol_id"]: verdict.molecule_A_scores.affinity_score, mol_b["mol_id"]: verdict.molecule_B_scores.affinity_score},
                    "druglikeness": {mol_a["mol_id"]: verdict.molecule_A_scores.admet_score, mol_b["mol_id"]: verdict.molecule_B_scores.admet_score},
                    "novelty": {mol_a["mol_id"]: verdict.molecule_A_scores.novelty_score, mol_b["mol_id"]: verdict.molecule_B_scores.novelty_score},
                },
                round_number=round_number,
            )

        except Exception as e:
            # 如果大模型崩溃，回退到原始的简易启发式打分 (避免中断整个管道)
            score_a = mol_a.get("docking_score", 0)
            score_b = mol_b.get("docking_score", 0)
            return TournamentMatch(
                match_id=match_id,
                mol_a=mol_a["mol_id"],
                mol_b=mol_b["mol_id"],
                winner=mol_a["mol_id"] if score_a < score_b else mol_b["mol_id"], # 对接分通常越负越好
                elo_shift=0.0,
                debate_summary=f"LLM API 异常，启用规则兜底: {str(e)}",
                dimensions={},
                round_number=round_number,
            )

    # -------------------------------------------------------------------------
    # Elo 系统与配对 (静态数学方法)
    # -------------------------------------------------------------------------
    @staticmethod
    def _elo_shift(rating_a: float, rating_b: float, k: float = 32.0) -> float:
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        return k * (1.0 - expected_a)

    @staticmethod
    def _swiss_pairing(
        molecules: List[MoleculeRecord],
        elo: Dict[MoleculeID, float],
    ) -> List[Tuple[MoleculeRecord, MoleculeRecord]]:
        sorted_mols = sorted(molecules, key=lambda m: elo.get(m["mol_id"], 1500.0), reverse=True)
        pairs = []
        for i in range(0, len(sorted_mols) - 1, 2):
            pairs.append((sorted_mols[i], sorted_mols[i + 1]))
        return pairs