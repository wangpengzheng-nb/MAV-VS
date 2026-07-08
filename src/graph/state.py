"""
AutoVS-Agent: 全局图状态定义 (Global Graph State) v2.0
========================================================
锦标赛机制重构版 — 基于 Co-Scientist 多智能体辩论 + Elo 闭环演化。

架构变化 (v1 → v2):
  旧: 线性 8 步漏斗 (Strategy → Clustering → ... → MetaReview)
  新: 侦察画像 → 多策略生成 → 红军辩论锦标赛 → Elo 排序 → 最佳策略进化

新增字段:
  - target_profile:       Target Scout 的靶点深度画像
  - candidate_strategies: 3-5 个差异化虚拟筛选策略
  - tournament_history:   红军辩论记录 + 裁判打分
  - best_strategy:        MetaReview 进化的最终胜出策略
"""

from __future__ import annotations

import uuid
import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from langgraph.graph.message import add_messages


# =============================================================================
# 基础类型别名
# =============================================================================

MoleculeID = str
SMILES = str
PDBPath = str


# =============================================================================
# 分子全生命周期数据容器 (保留 v1 兼容)
# =============================================================================

class MoleculeRecord(TypedDict, total=False):
    mol_id: MoleculeID
    smiles: SMILES
    source_db: str
    cluster_id: Optional[int]
    conformer_sdf_path: Optional[str]
    docking_score: Optional[float]
    docking_affinity: Optional[float]
    docking_pose_path: Optional[str]
    structural_score: Optional[float]
    pharmacophore_score: Optional[float]
    synthetic_accessibility: Optional[float]
    admet_flags: Optional[Dict[str, bool]]
    medchem_passed: Optional[bool]
    elo_rating: Optional[float]
    tournament_wins: Optional[int]
    tournament_losses: Optional[int]
    debate_rationale: Optional[str]
    md_dG: Optional[float]
    md_kd: Optional[float]
    md_hbond_occupancy: Optional[Dict[str, float]]
    md_rmsd_mean: Optional[float]
    md_passed: Optional[bool]
    mlp_pred_dG: Optional[float]
    mlp_uncertainty: Optional[float]


# =============================================================================
# 靶点蛋白信息 (保留 v1 兼容)
# =============================================================================

class TargetInfo(TypedDict):
    target_name: str
    uniprot_id: str
    pdb_id: str
    pdb_path: PDBPath
    binding_site_center: List[float]
    binding_site_size: List[float]
    key_residues: List[str]
    target_class: str
    description: str
    organism: str


# =============================================================================
# v2 新增: 靶点深度画像 (Target Scout 产出)
# =============================================================================

class StructuralAssessment(TypedDict):
    """靶点结构可用性评估。"""
    has_experimental_structure: bool
    pdb_ids: List[str]
    resolution_range: str          # "1.5-2.5A"
    has_cocrystal_with_ligand: bool
    pocket_type: str               # "deep_cleft" / "shallow_groove" / "flat_ppi" / "allosteric"
    pocket_volume_estimate: str    # "small (<300 A3)" / "medium (300-800)" / "large (>800)"
    pocket_polarity: str           # "hydrophobic" / "mixed" / "polar"
    flexibility_concern: str       # "rigid" / "moderate_flexibility" / "highly_flexible"


class KnownLigandInfo(TypedDict):
    """已知配体信息。"""
    has_known_active_ligands: bool
    representative_ligands: List[str]     # SMILES or names
    binding_affinity_range: str           # "nM" / "uM" / "mM"
    key_pharmacophore_features: List[str]
    relevant_patents_or_papers: List[str]  # PMID/DOI


class PriorityMetrics(TypedDict):
    """靶点特定的优先级评价指标。"""
    primary_metrics: List[str]     # 最重要的指标
    secondary_metrics: List[str]
    red_flags: List[str]           # 必须避免的特征
    suggested_thresholds: Dict[str, str]  # 建议的软阈值


class TargetProfile(TypedDict):
    """Target Scout 产出的完整靶点画像。"""
    target_name: str
    structural_assessment: StructuralAssessment
    known_ligand_info: KnownLigandInfo
    priority_metrics: PriorityMetrics
    drug_design_challenges: List[str]
    recommended_approaches: List[str]      # "SBDD" / "LBDD" / "FBDD" / "covalent" / "PROTAC"
    key_references: List[str]
    profile_timestamp: str


# =============================================================================
# v2 新增: 候选策略 (Strategy Generation 产出)
# =============================================================================

class FilterRule(TypedDict):
    """单条过滤规则。"""
    rule_id: str
    category: str                    # "absolute_filter" / "relative_ranking" / "soft_metric"
    description: str
    parameter: str                   # e.g. "MW", "LogP", "docking_score", "PAINS"
    operator: str                    # "<", ">", "<=", ">=", "in_range", "top_percentile"
    value: Any                       # threshold value or range
    rationale: str                   # scientific justification
    relaxable: bool                  # whether this can be relaxed in contingency
    relaxed_value: Optional[Any]     # relaxed threshold for contingency


class ContingencyPlan(TypedDict):
    """策略的应急预案。"""
    trigger_condition: str           # e.g. "survivors < 10"
    relaxation_steps: List[Dict[str, Any]]  # ordered list of {rule_id, new_value, reason}
    minimum_acceptable_thresholds: Dict[str, Any]
    fallback_strategy: str           # what to do if relaxation still fails


class CandidateStrategy(TypedDict):
    """单个虚拟筛选策略。"""
    strategy_name: str
    strategy_tagline: str            # one-line summary
    rationale: str                   # why this approach is suitable
    approach_type: str               # "structure_based" / "ligand_based" / "hybrid" / "ml_driven"
    absolute_filters: List[FilterRule]
    relative_rankings: List[FilterRule]
    soft_metrics: List[FilterRule]
    contingency_plan: ContingencyPlan
    estimated_survival_rate: str     # e.g. "~5% → ~5000 survivors from 100K"
    strengths: List[str]
    weaknesses: List[str]


# =============================================================================
# v2 新增: 锦标赛记录
# =============================================================================

class ExpertAttack(TypedDict):
    """单个红军队员的攻击意见。"""
    persona: str                     # "medchem_veteran" / "funnel_terminator" / "target_specialist"
    persona_name: str                # 可读名称
    attack_points: List[str]         # 攻击点列表
    severity: str                    # "critical" / "major" / "minor"
    suggested_fixes: List[str]
    agreement_with_strategy: float   # 0.0-1.0 对该策略的整体认可度


class DebateRound(TypedDict):
    """一轮完整的红蓝对抗辩论。"""
    round_id: str
    strategy_a: str                  # 策略名
    strategy_b: str                  # 策略名
    expert_attacks_on_a: List[ExpertAttack]   # 红军对策略A的攻击
    expert_attacks_on_b: List[ExpertAttack]   # 红军对策略B的攻击
    judge_summary: str               # 裁判的综合点评
    winner: str                      # 策略名 or "tie"
    elo_shift_a: float
    elo_shift_b: float
    key_deciding_factor: str         # 决定性因素
    timestamp: str


class TournamentState(TypedDict):
    """锦标赛运行状态。"""
    round_number: int
    max_rounds: int
    elo_ratings: Dict[str, float]           # strategy_name → Elo
    pairings_queue: List[List[str]]          # 待辩论的策略对
    completed_debates: int
    current_leader: str                      # 当前领先策略名
    elo_k_factor: float
    elo_initial_rating: float


# =============================================================================
# v1 保留类型 (向后兼容)
# =============================================================================

class DynamicFilterProtocol(TypedDict):
    mw_range: List[float]
    logp_range: List[float]
    hbd_max: int
    hba_max: int
    rotatable_bonds_max: int
    tpsa_range: List[float]
    num_aromatic_rings_range: List[int]
    pharmacophore_required: List[str]
    pharmacophore_optional: List[str]
    pharmacophore_excluded: List[str]
    excluded_substructures: List[str]
    toxic_groups: List[str]
    reactive_groups: List[str]
    docking_score_min: float
    rule_category: Literal["Ro5", "bRo5", "custom"]
    rationale: str
    literature_refs: List[str]
    version: int
    generated_at: str


class WatchdogConfig(TypedDict):
    grid_center: List[float]
    grid_size: List[float]
    exhaustiveness: int
    md_ensemble: str
    md_temperature: float
    md_simulation_time_ns: float
    md_force_field: str
    md_water_model: str
    dry_run_passed: bool
    positive_control_score: float
    decoy_rejection_rate: float
    error_log: List[str]


class MDSimulationRecord(TypedDict):
    mol_id: MoleculeID
    trajectory_path: str
    topology_path: str
    total_time_ns: float
    dG_mmgbsa: Optional[float]
    dG_mmpbsa: Optional[float]
    kd_predicted: Optional[float]
    ligand_rmsd_mean: float
    ligand_rmsd_std: float
    key_hbond_occupancy: Dict[str, float]
    protein_rmsd_mean: float
    complex_stable: bool
    simulation_status: Literal["completed", "failed", "crashed", "pending"]
    error_message: Optional[str]


class TournamentMatch(TypedDict):
    match_id: str
    mol_a: MoleculeID
    mol_b: MoleculeID
    winner: Optional[MoleculeID]
    elo_shift: float
    debate_summary: str
    dimensions: Dict[str, Dict[str, float]]
    round_number: int


class ClosedLoopKnowledge(TypedDict):
    total_iterations: int
    total_hits_found: int
    privileged_scaffolds: List[str]
    privileged_pharmacophores: List[Dict[str, Any]]
    favorable_interactions: List[str]
    unfavorable_patterns: List[str]
    md_derived_insights: List[str]
    false_positive_patterns: List[str]
    chemical_space_direction: Dict[str, float]
    recommended_scaffolds: List[str]
    convergence_trend: List[float]


class ActiveLearningState(TypedDict):
    iteration: int
    max_iterations: int
    convergence_threshold: float
    stagnation_counter: int
    best_dG_history: List[float]
    hit_rate_history: List[float]
    early_stop_patience: int
    acquisition_size: int
    elo_k_factor: float
    elo_initial_rating: float
    tournament_rounds: int
    top_n_to_md: int


class ExpertMember(TypedDict):
    expert_id: str
    name: str
    role_description: str
    tools: List[str]
    vote_weight: float


class ExpertCommitteeReport(TypedDict):
    expert_id: str
    batch_mol_ids: List[MoleculeID]
    scores: Dict[MoleculeID, float]
    flags: Dict[MoleculeID, List[str]]
    comments: Dict[MoleculeID, str]
    vetoed: List[MoleculeID]
    confidence: float


# =============================================================================
# 主图状态 (Master State) v2.0 — 锦标赛机制
# =============================================================================

class MACVSState(TypedDict):
    """AutoVS-Agent 主全局状态 (锦标赛重构版)。

    比 v1 新增 6 个字段:
      target_profile, candidate_strategies, tournament_history,
      best_strategy, tournament_state, strategy_generation_meta
    """

    # ---- 1. 会话元信息 ----
    session_id: str
    pipeline_stage: Literal[
        "init",
        "target_scout",           # Step 1: Target Scout 靶点侦察
        "strategy_generation",    # Step 2: 多策略生成
        "tournament",             # Step 3: 红军辩论锦标赛
        "meta_review",            # Step 4: 最佳策略进化
        # v1 兼容阶段
        "strategy", "clustering", "watchdog", "htvs",
        "medchem_filter", "ranking", "md_oracle",
        "converged", "error",
    ]
    created_at: str
    updated_at: str

    # ---- 2. 靶点信息 ----
    target_info: TargetInfo

    # ---- 3. v2 新增: 靶点画像 ----
    target_profile: Optional[TargetProfile]

    # ---- 4. v2 新增: 候选策略集 ----
    candidate_strategies: List[CandidateStrategy]

    # ---- 5. v2 新增: 锦标赛状态与记录 ----
    tournament_state: TournamentState
    tournament_history: List[DebateRound]

    # ---- 6. v2 新增: 最佳策略 ----
    best_strategy: Optional[dict]

    # ---- 7. v1 兼容: 过滤协议 ----
    filter_protocol: Optional[DynamicFilterProtocol]
    protocol_version: int

    # ---- 8. 分子库与候选池 ----
    full_library_path: str
    total_library_size: int
    candidate_pool: List[MoleculeRecord]
    surviving_pool: List[MoleculeRecord]
    screened_records: Dict[MoleculeID, MoleculeRecord]

    # ---- 9. 聚类 ----
    cluster_count: int
    cluster_centroids: Optional[List[List[float]]]
    cluster_method: str

    # ---- 10. Watchdog ----
    watchdog_config: Optional[WatchdogConfig]
    watchdog_retry_count: int
    watchdog_max_retries: int

    # ---- 11. HTVS ----
    htvs_total_docked: int
    htvs_job_id: Optional[str]
    htvs_completed: bool
    htvs_top_n: int

    # ---- 12. MedChem Committee ----
    committee_members: List[ExpertMember]
    committee_reports: List[ExpertCommitteeReport]
    committee_veto_threshold: float
    medchem_survivor_count: int

    # ---- 13. MPO 锦标赛 (分子级) ----
    elo_leaderboard: Dict[MoleculeID, float]
    tournament_bracket: List[TournamentMatch]
    mpo_survivor_count: int
    mpo_dimensions: List[str]

    # ---- 14. MD Oracle ----
    md_results: Dict[MoleculeID, MDSimulationRecord]
    md_job_ids: List[str]
    md_passed_count: int
    md_min_simulation_ns: float

    # ---- 15. 闭环知识库 ----
    knowledge_base: ClosedLoopKnowledge
    continue_next_iteration: Optional[bool]

    # ---- 16. Proxy MLP ----
    mlp_model_path: Optional[str]
    mlp_feature_dim: int
    mlp_ready: bool
    mlp_training_history: Optional[Dict[str, List[float]]]

    # ---- 17. 主动学习控制 ----
    al_state: ActiveLearningState

    # ---- 18. 最终产出 ----
    final_hits: List[MoleculeRecord]
    output_report_path: Optional[str]

    # ---- 19. 异常与日志 ----
    errors: Annotated[List[Dict[str, str]], operator.add]
    event_log: Annotated[List[str], operator.add]

    # ---- 20. LLM 消息历史 ----
    messages: Annotated[list, add_messages]


# =============================================================================
# 工厂函数
# =============================================================================

def create_initial_state(
    target_info: TargetInfo,
    full_library_path: str = "",
    total_library_size: int = 0,
    max_iterations: int = 5,
    htvs_top_n: int = 2000,
    medchem_target: int = 300,
    mpo_target: int = 20,
    md_target: int = 5,
    md_simulation_ns: float = 50.0,
    watchdog_max_retries: int = 3,
    elo_k_factor: float = 32.0,
    elo_initial_rating: float = 1500.0,
    tournament_rounds: int = 3,
    early_stop_patience: int = 2,
    convergence_threshold: float = 0.5,
    mlp_feature_dim: int = 2048,
    max_debate_rounds: int = 6,
) -> MACVSState:
    """创建锦标赛版初始状态。

    Args:
        target_info: 靶点蛋白信息。
        max_debate_rounds: 策略辩论最大轮数。
        其余参数保留 v1 兼容。
    """
    now = datetime.utcnow().isoformat()

    return MACVSState(
        # 1. 元信息
        session_id=str(uuid.uuid4()),
        pipeline_stage="init",
        created_at=now,
        updated_at=now,

        # 2. 靶点
        target_info=target_info,

        # 3-6. v2 新增
        target_profile=None,
        candidate_strategies=[],
        tournament_state=TournamentState(
            round_number=0,
            max_rounds=max_debate_rounds,
            elo_ratings={},
            pairings_queue=[],
            completed_debates=0,
            current_leader="",
            elo_k_factor=elo_k_factor,
            elo_initial_rating=elo_initial_rating,
        ),
        tournament_history=[],
        best_strategy=None,

        # 7. v1 兼容
        filter_protocol=None,
        protocol_version=0,

        # 8-18. v1 保留字段 (默认值)
        full_library_path=full_library_path,
        total_library_size=total_library_size,
        candidate_pool=[],
        surviving_pool=[],
        screened_records={},
        cluster_count=0,
        cluster_centroids=None,
        cluster_method="Butina",
        watchdog_config=None,
        watchdog_retry_count=0,
        watchdog_max_retries=watchdog_max_retries,
        htvs_total_docked=0,
        htvs_job_id=None,
        htvs_completed=False,
        htvs_top_n=htvs_top_n,
        committee_members=[],
        committee_reports=[],
        committee_veto_threshold=0.5,
        medchem_survivor_count=medchem_target,
        elo_leaderboard={},
        tournament_bracket=[],
        mpo_survivor_count=mpo_target,
        mpo_dimensions=["affinity", "druglikeness", "novelty"],
        md_results={},
        md_job_ids=[],
        md_passed_count=md_target,
        md_min_simulation_ns=md_simulation_ns,
        knowledge_base={},
        continue_next_iteration=None,
        mlp_model_path=None,
        mlp_feature_dim=mlp_feature_dim,
        mlp_ready=False,
        mlp_training_history=None,
        al_state=ActiveLearningState(
            iteration=0,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            stagnation_counter=0,
            best_dG_history=[],
            hit_rate_history=[],
            early_stop_patience=early_stop_patience,
            acquisition_size=0,
            elo_k_factor=elo_k_factor,
            elo_initial_rating=elo_initial_rating,
            tournament_rounds=tournament_rounds,
            top_n_to_md=mpo_target,
        ),
        final_hits=[],
        output_report_path=None,

        # 19.
        errors=[],
        event_log=[f"[{now}] Pipeline initialized v2.0. Target: {target_info.get('target_name')}"],

        # 20.
        messages=[],
    )


# =============================================================================
# 工具函数
# =============================================================================

def update_timestamp(state: MACVSState) -> Dict[str, str]:
    """返回时间戳更新 dict。"""
    return {"updated_at": datetime.utcnow().isoformat()}


def is_pipeline_complete(state: MACVSState) -> bool:
    """判断管道是否终止。"""
    return state.get("pipeline_stage") in ("converged", "error")


def is_converged(state: MACVSState) -> bool:
    """判断锦标赛是否收敛。"""
    ts = state.get("tournament_state", {})
    if ts.get("completed_debates", 0) >= ts.get("max_rounds", 6):
        return True
    return state.get("pipeline_stage") == "converged"


def get_top_elo(state: MACVSState, n: int = 5) -> List[tuple]:
    """按 Elo 降序获取 Top-N 策略。"""
    elo = state.get("tournament_state", {}).get("elo_ratings", {})
    return sorted(elo.items(), key=lambda x: x[1], reverse=True)[:n]
