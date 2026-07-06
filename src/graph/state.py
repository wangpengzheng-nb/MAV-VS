"""
AutoVS-Agent: 全局图状态定义 (Global Graph State)
==================================================
基于 LangGraph TypedDict 的 8 步闭环虚拟筛选漏斗状态机。

8 步漏斗流转:
  Step 1  Strategy        →  战前侦察，产出动态药化过滤协议
  Step 2  Clustering      →  化学空间降维，抽取代表分子 (~10万)
  Step 3  Watchdog        →  小样本演习，锁定对接/MD 参数
  Step 4  HTVS            →  高通量虚拟筛选粗筛 → Top 2000
  Step 5  MedChem Filter  →  绝对值淘汰 (PAINS/ADMET 一票否决) → Top 300
  Step 6  MPO Ranking     →  Elo 1v1 辩论锦标赛 → Top 20
  Step 7  MD Oracle       →  50ns MD + ΔG 终审 → 3-5 Hits
  Step 8  Meta-Review     →  优势特征提取 → 闭环回 Step 2 (下一轮迭代)

设计原则:
  - 不可变增量更新: 每个节点返回 dict partial update
  - 全链路可追溯: 每阶段产出物独立持久化
  - 类型安全: TypedDict + Annotated reducer
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

MoleculeID = str                        # 分子唯一标识符 (ZINC ID / 内部 UUID)
SMILES = str                            # SMILES 字符串
PDBPath = str                           # PDB 文件路径


# =============================================================================
# 分子全生命周期数据容器
# =============================================================================

class MoleculeRecord(TypedDict, total=False):
    """单条分子记录 —— 贯穿 8 步漏斗的原子数据单元。

    各步骤逐步填充不同字段:
      Step 1:  (策略阶段不操作分子)
      Step 2:  mol_id, smiles, source_db, cluster_id
      Step 3:  (Watchdog 只操作少量阳性/诱饵分子)
      Step 4:  docking_score, docking_affinity, docking_pose_path
      Step 5:  admet_flags, structural_score, pharmacophore_score, synthetic_accessibility
      Step 6:  elo_rating, tournament_wins, tournament_losses
      Step 7:  md_dG, md_kd, md_hbond_occupancy, md_rmsd, md_passed
      Step 8:  (Meta-Review 不修改单分子，更新全局知识库)
    """
    # ---- 身份 ----
    mol_id: MoleculeID
    smiles: SMILES
    source_db: str                       # "ZINC20" / "Enamine_REAL" / "ChemBL"
    cluster_id: Optional[int]            # Step 2 聚类分组 ID

    # ---- 3D 构象 ----
    conformer_sdf_path: Optional[str]

    # ---- Step 4: HTVS 对接 ----
    docking_score: Optional[float]       # GNINA CNNscore / CNN_VS
    docking_affinity: Optional[float]    # kcal/mol
    docking_pose_path: Optional[str]     # 最佳姿态 PDB 路径

    # ---- Step 5: MedChem 多维度 ----
    structural_score: Optional[float]    # 结构互补性 (PLIP)
    pharmacophore_score: Optional[float] # 药效团匹配
    synthetic_accessibility: Optional[float]  # SAscore
    admet_flags: Optional[Dict[str, bool]]    # {PAINS, BRENK, hERG_risk, CYP450_inhib, ...}
    medchem_passed: Optional[bool]       # 是否通过绝对值过滤

    # ---- Step 6: MPO 锦标赛 ----
    elo_rating: Optional[float]          # Elo 积分 (初始值 1500)
    tournament_wins: Optional[int]
    tournament_losses: Optional[int]
    debate_rationale: Optional[str]      # Ranking Agent 辩论摘要

    # ---- Step 7: MD Oracle ----
    md_dG: Optional[float]              # MM/GBSA 结合自由能 (kcal/mol)
    md_kd: Optional[float]              # 解离常数 (nM)
    md_hbond_occupancy: Optional[Dict[str, float]]  # {残基名: 占有率%}
    md_rmsd_mean: Optional[float]       # 配体 RMSD 均值 (Å)
    md_passed: Optional[bool]           # 是否通过 MD 终审

    # ---- Proxy MLP (可选加速路径) ----
    mlp_pred_dG: Optional[float]
    mlp_uncertainty: Optional[float]


# =============================================================================
# 靶点蛋白信息
# =============================================================================

class TargetInfo(TypedDict):
    """靶点结构/生化信息，由用户在管道启动时提供。"""
    target_name: str                     # "Bcl-2" / "Bcl-xl" / "EGFR"
    uniprot_id: str                      # "P10415"
    pdb_id: str                          # "60OK"
    pdb_path: PDBPath                    # 受体 PDB 本地路径
    binding_site_center: List[float]     # [x, y, z] (Å)
    binding_site_size: List[float]       # [sx, sy, sz] (Å)
    key_residues: List[str]              # ["ASP103", "TRP144", "GLY145"]
    target_class: str                    # "PPI" / "Kinase" / "GPCR" / "Protease"
    description: str                     # 靶点功能与疾病关联
    organism: str                        # "Homo sapiens"


# =============================================================================
# Step 1 产出: 动态药化过滤协议 (Strategy Agent)
# =============================================================================

class DynamicFilterProtocol(TypedDict):
    """Strategy Agent 产出的动态药化过滤规则手册。

    与静态 Lipinski 五规则不同，本协议根据靶点特征动态调整阈值。
    例如 PPI 大口袋靶点可使用 bRo5 (beyond Rule-of-5) 宽松规则。
    """
    # ---- 物理化学规则 (动态阈值) ----
    mw_range: List[float]                # [min, max] Da
    logp_range: List[float]              # [min, max]
    hbd_max: int
    hba_max: int
    rotatable_bonds_max: int
    tpsa_range: List[float]
    num_aromatic_rings_range: List[int]

    # ---- 药效团需求 ----
    pharmacophore_required: List[str]    # 必需特征: ["hydrophobic_ring", "hbond_acceptor"]
    pharmacophore_optional: List[str]    # 加分特征
    pharmacophore_excluded: List[str]

    # ---- 子结构黑名单 ----
    excluded_substructures: List[str]    # PAINS / BRENK 子结构 SMARTS
    toxic_groups: List[str]              # 毒性基团 SMARTS
    reactive_groups: List[str]           # 反应性基团 SMARTS

    # ---- 对接评分阈值 ----
    docking_score_min: float             # 最低对接分数门槛

    # ---- 元信息 ----
    rule_category: Literal["Ro5", "bRo5", "custom"]  # 规则类别
    rationale: str                       # 科学依据
    literature_refs: List[str]           # 参考文献 PMID/DOI
    version: int
    generated_at: str                    # ISO 时间戳


# =============================================================================
# Step 3 产出: Watchdog 锁定参数
# =============================================================================

class WatchdogConfig(TypedDict):
    """Watchdog Agent 在小样本演习后锁定的计算参数。

    这些参数将冻结并在 Step 4 (HTVS) 和 Step 7 (MD) 中使用。
    """
    # ---- 对接盒子 (Grid Box) ----
    grid_center: List[float]             # [x, y, z] — 经 Watchdog 纠偏后的中心
    grid_size: List[float]               # [sx, sy, sz] — 最优盒子尺寸
    exhaustiveness: int                  # GNINA 穷举度 (8-64)

    # ---- MD 参数 ----
    md_ensemble: str                     # "NPT" / "NVT"
    md_temperature: float                # K (通常 300/310)
    md_simulation_time_ns: float         # 模拟时长 (ns)
    md_force_field: str                  # "amber14sb" / "charmm36"
    md_water_model: str                  # "tip3p" / "spc"

    # ---- 演习结果 ----
    dry_run_passed: bool                 # 演习是否通过
    positive_control_score: float        # 阳性对照对接分数
    decoy_rejection_rate: float          # 诱饵排除率
    error_log: List[str]                 # 演习中遇到的异常及修复记录


# =============================================================================
# Step 7 产出: MD Oracle 结果
# =============================================================================

class MDSimulationRecord(TypedDict):
    """单条 MD 模拟的详细记录。"""
    mol_id: MoleculeID
    trajectory_path: str                 # 轨迹文件路径 (.xtc / .dcd)
    topology_path: str                   # 拓扑文件路径
    total_time_ns: float                 # 实际模拟时长
    dG_mmgbsa: Optional[float]           # MM/GBSA 结合自由能 (kcal/mol)
    dG_mmpbsa: Optional[float]           # MM/PBSA 结合自由能 (kcal/mol)
    kd_predicted: Optional[float]        # 预测 Kd (nM)
    ligand_rmsd_mean: float              # 配体 RMSD 均值
    ligand_rmsd_std: float               # 配体 RMSD 标准差
    key_hbond_occupancy: Dict[str, float] # 关键氢键占有率
    protein_rmsd_mean: float             # 蛋白骨架 RMSD
    complex_stable: bool                 # 复合物是否稳定 (RMSD < 3Å)
    simulation_status: Literal["completed", "failed", "crashed", "pending"]
    error_message: Optional[str]


# =============================================================================
# Step 6 辅助: 锦标赛对阵记录
# =============================================================================

class TournamentMatch(TypedDict):
    """一场 1v1 辩论的完整记录。"""
    match_id: str
    mol_a: MoleculeID
    mol_b: MoleculeID
    winner: Optional[MoleculeID]         # 胜者 mol_id，None 为平局
    elo_shift: float                     # Elo 积分变化量
    debate_summary: str                  # Ranking Agent 的裁决摘要
    dimensions: Dict[str, Dict[str, float]]  # {affinity: {mol_a: x, mol_b: y}, ...}
    round_number: int


# =============================================================================
# 闭环知识库 (Step 8 产出，反馈至 Step 2)
# =============================================================================

class ClosedLoopKnowledge(TypedDict):
    """Meta-Review Agent 维护的闭环累积知识库。

    每一轮迭代都会更新此知识库，提炼 MD 验证命中的共性特征，
    指导下一轮聚类和筛选的方向。
    """
    # ---- 累积统计 ----
    total_iterations: int
    total_hits_found: int

    # ---- 赢家化学空间特征 ----
    privileged_scaffolds: List[str]       # 优势骨架 SMILES
    privileged_pharmacophores: List[Dict[str, Any]]  # 优势药效团模式
    favorable_interactions: List[str]     # 有利相互作用模式 (如 π-π stacking)
    unfavorable_patterns: List[str]       # 不利子结构模式

    # ---- 教训 ----
    md_derived_insights: List[str]        # 从 MD 轨迹中提取的动力学教训
    false_positive_patterns: List[str]    # 对接高分但 MD 失败的假阳性模式

    # ---- 方向指引 ----
    chemical_space_direction: Dict[str, float]  # 下一轮化学空间偏移向量
    recommended_scaffolds: List[str]      # 推荐下一轮富集的骨架
    convergence_trend: List[float]        # 每轮最佳 ΔG 趋势


# =============================================================================
# 主动学习循环控制
# =============================================================================

class ActiveLearningState(TypedDict):
    """闭环主动学习的内部状态追踪。"""
    iteration: int                        # 当前闭环轮次 (从 0 开始)
    max_iterations: int                   # 最大迭代轮次
    convergence_threshold: float          # ΔG 改进收敛阈值 (kcal/mol)
    stagnation_counter: int               # 连续未改进轮次
    best_dG_history: List[float]          # 每轮最佳 ΔG 历史 (越负越好)
    hit_rate_history: List[float]         # 每轮命中率历史 (MD passed/total)
    early_stop_patience: int              # 早停耐心轮次
    acquisition_size: int                 # 每轮从大库新增采样分子数

    # ---- Elo 赛制配置 ----
    elo_k_factor: float                   # Elo K 因子 (默认 32)
    elo_initial_rating: float             # 初始 Elo 分 (默认 1500)
    tournament_rounds: int                # 锦标赛轮次
    top_n_to_md: int                      # 进入 MD 的分子数 (默认 20)


# =============================================================================
# 专家委员会成员定义
# =============================================================================

class ExpertMember(TypedDict):
    """MedChem Committee 中的单个专家成员元信息。"""
    expert_id: str                        # "structural_biologist" / "admet_tox" / "medchem_synthesis"
    name: str                             # 可读名称
    role_description: str                 # 职责描述
    tools: List[str]                      # 可调用工具列表
    vote_weight: float                    # 投票权重 [0, 1]


class ExpertCommitteeReport(TypedDict):
    """MedChem Committee 对一批分子的集体评估报告。"""
    expert_id: str
    batch_mol_ids: List[MoleculeID]
    scores: Dict[MoleculeID, float]       # mol_id → 评分
    flags: Dict[MoleculeID, List[str]]    # mol_id → 警告列表
    comments: Dict[MoleculeID, str]       # mol_id → 专家评语
    vetoed: List[MoleculeID]              # 一票否决的分子 ID 列表
    confidence: float                     # [0, 1]


# =============================================================================
# 主图状态 (Master State) —— LangGraph 核心
# =============================================================================

class MACVSState(TypedDict):
    """AutoVS-Agent 8 步漏斗的主全局状态。

    这是 LangGraph StateGraph 中所有节点共享的唯一状态对象。
    每个节点接收 MACVSState，返回 dict partial update。
    """

    # =========================================================================
    # 1. 会话元信息
    # =========================================================================
    session_id: str
    pipeline_stage: Literal[
        "init",               # 初始化
        "strategy",           # Step 1: Strategy Agent 生成过滤协议
        "clustering",         # Step 2: 化学空间聚类降维
        "watchdog",           # Step 3: Watchdog 小样本演习 + 参数锁定
        "htvs",               # Step 4: 高通量虚拟筛选
        "medchem_filter",     # Step 5: MedChem Committee 绝对值淘汰
        "ranking",            # Step 6: MPO Elo 锦标赛排序
        "md_oracle",          # Step 7: MD 模拟终极验证
        "meta_review",        # Step 8: Meta-Review 闭环复盘
        "converged",          # 管道收敛
        "error",              # 异常终止
    ]
    created_at: str
    updated_at: str

    # =========================================================================
    # 2. 靶点与策略 (Step 1)
    # =========================================================================
    target_info: TargetInfo
    filter_protocol: Optional[DynamicFilterProtocol]  # Step 1 产出
    protocol_version: int                             # 策略版本号 (迭代更新时递增)

    # =========================================================================
    # 3. 分子库与候选池
    # =========================================================================
    full_library_path: str                           # 大库源文件路径
    total_library_size: int                          # 全库分子总数
    candidate_pool: List[MoleculeRecord]             # 当前轮次候选池 (~10万 → 逐渐淘汰)
    surviving_pool: List[MoleculeRecord]             # 当前存活的分子 (经各步过滤后)
    screened_records: Dict[MoleculeID, MoleculeRecord]  # 所有已完成评估的分子归档

    # =========================================================================
    # 4. Step 2: 聚类快照
    # =========================================================================
    cluster_count: int                               # 聚类簇数
    cluster_centroids: Optional[List[List[float]]]   # 各簇中心坐标 (用于定向挖掘)
    cluster_method: str                              # "Butina"/"K-Means"/"DBSCAN"

    # =========================================================================
    # 5. Step 3: Watchdog 参数
    # =========================================================================
    watchdog_config: Optional[WatchdogConfig]
    watchdog_retry_count: int                        # 演习重试次数
    watchdog_max_retries: int                        # 最大重试次数 (默认 3)

    # =========================================================================
    # 6. Step 4: HTVS 执行
    # =========================================================================
    htvs_total_docked: int                           # 实际对接分子数
    htvs_job_id: Optional[str]                       # Slurm 作业 ID
    htvs_completed: bool
    htvs_top_n: int                                  # HTVS 保留数 (默认 2000)

    # =========================================================================
    # 7. Step 5: MedChem Committee
    # =========================================================================
    committee_members: List[ExpertMember]            # 专家委员会成员定义
    committee_reports: List[ExpertCommitteeReport]   # 委员会评估报告
    committee_veto_threshold: float                  # 否决票数阈值
    medchem_survivor_count: int                      # 过滤后存活分子数 (目标 300)

    # =========================================================================
    # 8. Step 6: MPO 锦标赛
    # =========================================================================
    elo_leaderboard: Dict[MoleculeID, float]         # Elo 积分榜
    tournament_bracket: List[TournamentMatch]        # 锦标赛对阵记录
    mpo_survivor_count: int                          # 锦标赛存活分子数 (目标 20)
    mpo_dimensions: List[str]                        # 评分维度: ["affinity", "druglikeness", "novelty"]

    # =========================================================================
    # 9. Step 7: MD Oracle
    # =========================================================================
    md_results: Dict[MoleculeID, MDSimulationRecord] # MD 模拟结果集
    md_job_ids: List[str]                            # Slurm 作业 ID 列表
    md_passed_count: int                             # 通过 MD 的分子数 (目标 3-5)
    md_min_simulation_ns: float                      # 最低 MD 模拟时长 (默认 50ns)

    # =========================================================================
    # 10. Step 8: 闭环知识库
    # =========================================================================
    knowledge_base: ClosedLoopKnowledge              # 累积先验知识
    continue_next_iteration: Optional[bool]          # Meta-Review 决定是否继续下一轮

    # =========================================================================
    # 11. Proxy MLP (可选加速路径)
    # =========================================================================
    mlp_model_path: Optional[str]
    mlp_feature_dim: int                             # 2048 (ECFP4)
    mlp_ready: bool
    mlp_training_history: Optional[Dict[str, List[float]]]

    # =========================================================================
    # 12. 主动学习控制
    # =========================================================================
    al_state: ActiveLearningState

    # =========================================================================
    # 13. 最终产出
    # =========================================================================
    final_hits: List[MoleculeRecord]                 # 最终命中的 3-5 个分子
    output_report_path: Optional[str]

# =========================================================================
    # 14. 异常与日志
    # =========================================================================
    errors: Annotated[List[Dict[str, str]], operator.add]   # 修改：使用 operator.add 保证追加而不覆盖
    event_log: Annotated[List[str], operator.add]           # 修改：使用 operator.add 保证追加而不覆盖

    # =========================================================================
    # 15. LLM 消息历史 (LangGraph 标准字段, add_messages reducer)
    # =========================================================================
    messages: Annotated[list, add_messages]


# =============================================================================
# 工厂函数: 创建管道初始状态
# =============================================================================

def create_initial_state(
    target_info: TargetInfo,
    full_library_path: str,
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
) -> MACVSState:
    """创建 AutoVS-Agent 8 步漏斗的初始状态快照。

    Args:
        target_info: 靶点蛋白信息。
        full_library_path: 全量分子库文件路径 (.smi / .sdf)。
        total_library_size: 分子库总数（0 表示后续自动统计）。
        max_iterations: 闭环最大迭代轮次。
        htvs_top_n: HTVS 粗筛保留数。
        medchem_target: MedChem 过滤后目标分子数。
        mpo_target: 锦标赛后目标分子数。
        md_target: MD 后目标命中分子数。
        md_simulation_ns: 每分子 MD 模拟时长 (ns)。
        watchdog_max_retries: Watchdog 演习最大重试次数。
        elo_k_factor: Elo K 因子。
        elo_initial_rating: Elo 初始积分。
        tournament_rounds: 锦标赛轮次。
        early_stop_patience: 早停耐心值。
        convergence_threshold: ΔG 收敛阈值 (kcal/mol)。
        mlp_feature_dim: MLP 指纹维度。

    Returns:
        完整的 MACVSState 初始状态字典。
    """
    now = datetime.utcnow().isoformat()

    return MACVSState(
        # 1. 元信息
        session_id=str(uuid.uuid4()),
        pipeline_stage="init",
        created_at=now,
        updated_at=now,

        # 2. 靶点与策略
        target_info=target_info,
        filter_protocol=None,
        protocol_version=0,

        # 3. 分子库
        full_library_path=full_library_path,
        total_library_size=total_library_size,
        candidate_pool=[],
        surviving_pool=[],
        screened_records={},

        # 4. 聚类
        cluster_count=0,
        cluster_centroids=None,
        cluster_method="Butina",

        # 5. Watchdog
        watchdog_config=None,
        watchdog_retry_count=0,
        watchdog_max_retries=watchdog_max_retries,

        # 6. HTVS
        htvs_total_docked=0,
        htvs_job_id=None,
        htvs_completed=False,
        htvs_top_n=htvs_top_n,

        # 7. MedChem Committee
        committee_members=_default_committee(),
        committee_reports=[],
        committee_veto_threshold=0.5,
        medchem_survivor_count=medchem_target,

        # 8. MPO 锦标赛
        elo_leaderboard={},
        tournament_bracket=[],
        mpo_survivor_count=mpo_target,
        mpo_dimensions=["affinity", "druglikeness", "novelty"],

        # 9. MD Oracle
        md_results={},
        md_job_ids=[],
        md_passed_count=md_target,
        md_min_simulation_ns=md_simulation_ns,

        # 10. 闭环知识库
        knowledge_base=_default_knowledge_base(),
        continue_next_iteration=None,

        # 11. Proxy MLP
        mlp_model_path=None,
        mlp_feature_dim=mlp_feature_dim,
        mlp_ready=False,
        mlp_training_history=None,

        # 12. 主动学习
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

        # 13. 最终产出
        final_hits=[],
        output_report_path=None,

        # 14. 异常与日志
        errors=[],
        event_log=[f"[{now}] Pipeline initialized. Target: {target_info.get('target_name')}, Library: {full_library_path}"],

        # 15. 消息
        messages=[],
    )


def _default_committee() -> List[ExpertMember]:
    """生成默认的 MedChem 三专家委员会配置。"""
    return [
        ExpertMember(
            expert_id="structural_biologist",
            name="结构生物学家",
            role_description="评估配体-受体结构互补性: 氢键网络、疏水匹配、空间位阻",
            tools=["PLIP", "PyMOL", "ProLIF"],
            vote_weight=0.4,
        ),
        ExpertMember(
            expert_id="admet_specialist",
            name="ADMET/毒理专家",
            role_description="评估吸收、分布、代谢、排泄、毒性: PAINS、BRENK、hERG、CYP450",
            tools=["RDKit", "SwissADME", "ADMETlab"],
            vote_weight=0.35,
        ),
        ExpertMember(
            expert_id="medchem_synthesis",
            name="药物化学/合成专家",
            role_description="评估合成可及性、结构新颖性、SAR 潜力",
            tools=["RDKit", "SAscore", "SCScore"],
            vote_weight=0.25,
        ),
    ]


def _default_knowledge_base() -> ClosedLoopKnowledge:
    """生成空的闭环知识库。"""
    return ClosedLoopKnowledge(
        total_iterations=0,
        total_hits_found=0,
        privileged_scaffolds=[],
        privileged_pharmacophores=[],
        favorable_interactions=[],
        unfavorable_patterns=[],
        md_derived_insights=[],
        false_positive_patterns=[],
        chemical_space_direction={},
        recommended_scaffolds=[],
        convergence_trend=[],
    )


# =============================================================================
# 状态检查工具函数
# =============================================================================

def is_pipeline_complete(state: MACVSState) -> bool:
    """判断管道是否已终止。"""
    return state["pipeline_stage"] in ("converged", "error")


def is_converged(state: MACVSState) -> bool:
    """判断闭环是否已收敛。"""
    al = state["al_state"]
    if al["iteration"] >= al["max_iterations"]:
        return True
    if al["stagnation_counter"] >= al["early_stop_patience"]:
        return True
    # 如果连续两轮 ΔG 改进 < 收敛阈值
    dg = al["best_dG_history"]
    if len(dg) >= 2 and abs(dg[-1] - dg[-2]) < al["convergence_threshold"]:
        return True
    return False


def get_survivor_count(state: MACVSState) -> int:
    """返回当前漏斗中存活的分子数。"""
    return len(state["surviving_pool"])


def get_top_elo(state: MACVSState, n: int = 20) -> List[tuple[MoleculeID, float]]:
    """按 Elo 积分降序获取 Top-N 分子。"""
    sorted_elo = sorted(
        state["elo_leaderboard"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return sorted_elo[:n]


def get_md_passed_hits(state: MACVSState) -> List[MoleculeRecord]:
    """获取通过 MD Oracle 验证的分子列表。"""
    return [m for m in state["surviving_pool"] if m.get("md_passed")]


def update_timestamp(state: MACVSState) -> Dict[str, str]:
    """返回时间戳更新 dict，供各节点调用。"""
    return {"updated_at": datetime.utcnow().isoformat()}


def export_checkpoint(state: MACVSState) -> Dict[str, Any]:
    """导出轻量级状态快照，用于断点续跑。"""
    al = state["al_state"]
    return {
        "session_id": state["session_id"],
        "pipeline_stage": state["pipeline_stage"],
        "iteration": al["iteration"],
        "protocol_version": state["protocol_version"],
        "survivor_count": get_survivor_count(state),
        "top_elo": get_top_elo(state, n=5),
        "best_dG": al["best_dG_history"][-1] if al["best_dG_history"] else None,
        "md_passed": get_md_passed_hits(state).__len__(),
        "error_count": len(state["errors"]),
        "updated_at": state["updated_at"],
    }
