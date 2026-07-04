"""
MAC-VS V2.0 全局图状态定义 (Global Graph State)
=================================================
基于 LangGraph 的 TypedDict 状态机，串联 Target Scout → Expert Committee →
Judge Agent → Proxy MLP 的闭环主动学习管道。

设计原则:
  - 不可变性: 每个节点返回状态的增量更新 (dict partial update)
  - 可追溯性: 每个阶段产出物独立存储，支持断点续跑和审计
  - 类型安全: 使用 TypedDict + Annotated reducer 确保字段类型稳定
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from langgraph.graph.message import add_messages


# =============================================================================
# 基础类型别名
# =============================================================================

# 分子唯一标识符 (如 ZINC ID、ChemBL ID、ChEMBL ID 或内部 UUID)
MoleculeID = str

# SMILES 字符串
SMILES = str

# PDB 格式的受体/配体文件路径或内容
PDBPath = str


# =============================================================================
# 分子数据模型
# =============================================================================

class MoleculeRecord(TypedDict):
    """单条分子记录 —— 贯穿整个管道的原子数据单元。

    不同阶段会逐步填充不同字段:
      - 入库时:     mol_id, smiles, source_db
      - 对接后:     docking_score, docking_pose_path
      - 专家评估后:  admet_flags, pharm_scores
      - 法官打分后:  judge_score, judge_labels
      - MLP 预测后:  mlp_pred_dG, mlp_uncertainty
    """
    # ---- 身份信息 ----
    mol_id: MoleculeID
    smiles: SMILES
    source_db: str                         # e.g. "ZINC20", "Enamine_REAL", "ChemBL"

    # ---- 3D 构象 / 对接 ----
    conformer_sdf_path: Optional[str]       # RDKit 生成的 3D SDF 路径
    docking_score: Optional[float]          # GNINA/smina 对接分数 (CNNscore / CNN_VS)
    docking_pose_path: Optional[PDBPath]    # 最佳对接姿态 PDB 路径
    docking_affinity: Optional[float]       # 对接亲和力 (kcal/mol)

    # ---- 专家委员会评估 (Stage 2) ----
    structural_score: Optional[float]       # 结构互补性评分 (PLIP 氢键/疏水/盐桥)
    admet_flags: Optional[Dict[str, bool]]  # ADMET 五规则标记 {PAINS: bool, BRENK: bool, ...}
    pharmacophore_score: Optional[float]    # 药效团匹配评分
    synthetic_accessibility: Optional[float] # 合成可及性 (SAscore)

    # ---- 法官裁决 (Stage 2 End) ----
    judge_score: Optional[float]            # 综合 ΔG 预测 (回归主任务)
    judge_labels: Optional[Dict[str, bool]] # 4 布尔诊断标签
                                            # {bbb_penetrant, hERG_blocker,
                                            #  cytotoxicity, solubility_issue}
    judge_rationale: Optional[str]          # 法官决策理由

    # ---- Proxy MLP (Stage 3) ----
    mlp_pred_dG: Optional[float]            # MLP 预测的结合自由能
    mlp_uncertainty: Optional[float]        # 预测不确定性 (用于主动学习采样)
    mlp_labels: Optional[Dict[str, float]]  # MLP 预测的 4 标签概率

    # ---- MD 最终验证 ----
    md_passed: Optional[bool]               # 是否通过 MD 模拟验证
    md_dG_exp: Optional[float]              # MD 计算的实验级 ΔG


# =============================================================================
# 靶点信息
# =============================================================================

class TargetInfo(TypedDict):
    """靶点蛋白质的结构与生化信息，由用户在管道启动时提供。"""
    target_name: str                        # e.g. "Bcl-2", "Bcl-xl"
    uniprot_id: str                         # e.g. "P10415"
    pdb_id: str                             # e.g. "60OK"
    pdb_path: PDBPath                       # 受体 PDB 文件本地路径
    binding_site_center: List[float]        # [x, y, z] 对接盒子中心 (Å)
    binding_site_size: List[float]          # [sx, sy, sz] 对接盒子尺寸 (Å)
    key_residues: Optional[List[str]]       # 关键结合位点残基 e.g. ["ASP103", "TRP144", "GLY145"]
    description: str                        # 靶点功能与疾病关联描述


# =============================================================================
# 筛选规则手册 (Target Scout 产出)
# =============================================================================

class Rulebook(TypedDict):
    """Stage 1 Target Scout 代理生成的动态筛选规则手册。

    包含多维度过滤条件，支持硬规则(一票否决)和软规则(容差区间)。
    """
    # 物理化学规则
    mw_range: List[float]                   # 分子量范围 [min, max] (Da)
    logp_range: List[float]                 # LogP 范围 [min, max]
    hbd_max: int                            # 氢键供体上限
    hba_max: int                            # 氢键受体上限
    rotatable_bonds_max: int                # 可旋转键上限
    tpsa_range: List[float]                 # TPSA 范围 [min, max]

    # 药效团规则
    pharmacophore_features: List[str]        # 必需药效团特征 e.g. ["hydrophobic_ring", "hbond_donor"]
    excluded_substructures: List[str]        # PAINS / 毒性子结构黑名单 (SMARTS)

    # 结构规则 (来自 PLIP 分析或文献)
    required_interactions: List[Dict[str, Any]]  # e.g. [{"type":"hbond","residue":"ASP103","tolerance":0.3}]

    # 元信息
    rationale: str                          # Target Scout 产出此规则的文献/知识依据
    generation_timestamp: str               # ISO 时间戳


# =============================================================================
# 专家委员会报告 (Stage 2 产出)
# =============================================================================

class ExpertReport(TypedDict):
    """单个专家代理对一批分子的评估报告。"""
    expert_name: str                        # 专家名称 e.g. "Structural_Biologist", "ADMET_Specialist"
    expert_role: Literal["structural", "admet", "pharmacophore", "synthesis"]
    batch_mol_ids: List[MoleculeID]         # 评估的分子 ID 列表
    scores: Dict[MoleculeID, float]         # mol_id → 专家评分
    flags: Optional[Dict[MoleculeID, List[str]]]  # mol_id → 警告/标记列表
    comments: Dict[MoleculeID, str]         # mol_id → 专家评语
    confidence: float                       # 专家对本次评估的置信度 [0,1]


# =============================================================================
# 主动学习循环控制
# =============================================================================

class ActiveLearningState(TypedDict):
    """主动学习循环的内部状态追踪。"""
    iteration: int                          # 当前迭代轮次 (从 0 开始)
    max_iterations: int                     # 最大迭代次数
    convergence_threshold: float            # 分数收敛阈值
    uncertainty_threshold: float            # 不确定性采样阈值
    stagnation_counter: int                 # 连续未改进轮次计数
    best_score_history: List[float]         # 每轮最佳分数的历史记录
    acquisition_batch_size: int             # 每轮采集的分子数量


# =============================================================================
# 主图状态 (Master State) —— LangGraph 核心
# =============================================================================

class MACVSState(TypedDict):
    """MAC-VS 管道的主全局状态。

    这是 LangGraph 图中所有节点共享的唯一状态对象。每个节点接收当前
    MACVSState，返回部分更新 dict。

    LangGraph 的 `Annotated[list, add_messages]` reducer 确保消息列表
    在各节点间以追加而非覆盖的方式合并。
    """

    # =========================================================================
    # 1. 会话元信息
    # =========================================================================
    session_id: str                         # UUID v4 会话标识符
    pipeline_stage: Literal[
        "init",                             # 初始化
        "scouting",                         # Stage 1: Target Scout 生成规则
        "screening",                        # Stage 2: 专家委员会 + 法官打分
        "training",                         # Stage 3: Proxy MLP 训练
        "predicting",                       # Stage 3: MLP 大规模预测
        "validating",                       # Final: MD 湿实验验证
        "converged",                        # 管道收敛，输出最终结果
        "error",                            # 异常终止
    ]
    created_at: str                         # 管道启动 ISO 时间戳
    updated_at: str                         # 最后节点写入时间戳

    # =========================================================================
    # 2. 靶点与规则
    # =========================================================================
    target_info: TargetInfo                 # 用户输入的目标蛋白信息
    rulebook: Optional[Rulebook]            # Stage 1 产出：动态规则手册
    rulebook_version: int                   # 规则手册版本号（收敛循环时 rulebook 可能被修订）

    # =========================================================================
    # 3. 分子库
    # =========================================================================
    full_library_path: str                  # 大规模分子库的源文件路径 (.smi / .sdf)
    total_library_size: int                 # 全库分子总数
    candidate_pool: List[MoleculeRecord]    # 当前候选池（经初筛后的备选分子）
    screened_records: Dict[MoleculeID, MoleculeRecord]  # 已完成全流程的分子记录

    # =========================================================================
    # 4. 当前批次 (Current Iteration)
    # =========================================================================
    current_batch: List[MoleculeRecord]     # 当前迭代正在处理的分子批次
    expert_reports: List[ExpertReport]      # 本轮所有专家的评估报告

    # =========================================================================
    # 5. 主动学习控制
    # =========================================================================
    al_state: ActiveLearningState           # AL 循环状态追踪器

    # =========================================================================
    # 6. Proxy MLP 模型状态
    # =========================================================================
    mlp_model_path: Optional[str]           # 训练好的 MLP 模型权重路径
    mlp_training_history: Optional[Dict[str, List[float]]]  # {loss: [...], val_loss: [...], ...}
    mlp_feature_dim: int                    # MLP 输入特征维度 (分子指纹长度)
    mlp_ready: bool                         # MLP 是否完成训练可投入推理

    # =========================================================================
    # 7. 最终产出
    # =========================================================================
    ranked_hits: List[MoleculeRecord]       # 最终排序后的命中分子列表 (Top-N)
    md_validated_hits: List[MoleculeRecord] # 通过 MD 模拟验证的分子
    output_report_path: Optional[str]       # 最终报告导出路径

    # =========================================================================
    # 8. 异常与日志
    # =========================================================================
    errors: List[Dict[str, str]]            # 错误堆栈 [{node, timestamp, message}]
    log: List[str]                          # 关键事件日志

    # =========================================================================
    # 9. LLM 消息历史 (LangGraph 标准字段)
    #    使用 add_messages reducer 实现跨节点消息自动归并
    # =========================================================================
    messages: Annotated[list, add_messages]


# =============================================================================
# 工厂函数 —— 为管道启动提供干净的状态快照
# =============================================================================

def create_initial_state(
    target_info: TargetInfo,
    full_library_path: str,
    total_library_size: int = 0,
    max_iterations: int = 10,
    acquisition_batch_size: int = 500,
    mlp_feature_dim: int = 2048,
    convergence_threshold: float = 0.01,
    uncertainty_threshold: float = 0.15,
) -> MACVSState:
    """创建管道的初始状态。

    Args:
        target_info: 靶点蛋白信息，由用户填入。
        full_library_path: 全量分子库 SDF/SMI 文件路径。
        total_library_size: 分子库总数（若未知填 0，后续由 loader 节点统计）。
        max_iterations: 主动学习最大迭代轮数。
        acquisition_batch_size: 每轮主动学习采集分子数。
        mlp_feature_dim: MLP 输入特征维度 (Morgan/ECFP 指纹长度)。
        convergence_threshold: 分数收敛判断阈值。
        uncertainty_threshold: 不确定性采样触发阈值。

    Returns:
        符合 MACVSState 定义的初始状态字典。
    """
    now = datetime.utcnow().isoformat()

    return MACVSState(
        # 会话元信息
        session_id=str(uuid.uuid4()),
        pipeline_stage="init",
        created_at=now,
        updated_at=now,

        # 靶点与规则
        target_info=target_info,
        rulebook=None,
        rulebook_version=0,

        # 分子库
        full_library_path=full_library_path,
        total_library_size=total_library_size,
        candidate_pool=[],
        screened_records={},

        # 当前批次
        current_batch=[],
        expert_reports=[],

        # 主动学习
        al_state=ActiveLearningState(
            iteration=0,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            uncertainty_threshold=uncertainty_threshold,
            stagnation_counter=0,
            best_score_history=[],
            acquisition_batch_size=acquisition_batch_size,
        ),

        # Proxy MLP
        mlp_model_path=None,
        mlp_training_history=None,
        mlp_feature_dim=mlp_feature_dim,
        mlp_ready=False,

        # 最终产出
        ranked_hits=[],
        md_validated_hits=[],
        output_report_path=None,

        # 异常与日志
        errors=[],
        log=[f"[{now}] Pipeline initialized for target: {target_info.get('target_name', 'Unknown')}"],

        # 消息
        messages=[],
    )


# =============================================================================
# 状态检查辅助函数
# =============================================================================

def is_pipeline_complete(state: MACVSState) -> bool:
    """判断管道是否已完成（收敛 或 超过最大迭代次数）。"""
    al = state["al_state"]
    return (
        state["pipeline_stage"] == "converged"
        or al["iteration"] >= al["max_iterations"]
    )


def get_unlabeled_count(state: MACVSState) -> int:
    """返回尚未经法官打分的候选分子数量。"""
    return sum(
        1 for mol in state["candidate_pool"]
        if mol["judge_score"] is None
    )


def get_top_hits(state: MACVSState, n: int = 10) -> List[MoleculeRecord]:
    """按法官综合评分获取 Top-N 命中分子。

    如果 judge_score 不可用，回退到 docking_score。
    """
    scored = [
        m for m in state["candidate_pool"]
        if m["judge_score"] is not None or m["docking_score"] is not None
    ]
    scored.sort(
        key=lambda m: m.get("judge_score") or m.get("docking_score") or 0.0,
        reverse=True,  # 分数越高越好
    )
    return scored[:n]


# =============================================================================
# 状态快照导出 (用于 checkpoint / 断点续跑)
# =============================================================================

def export_checkpoint(state: MACVSState) -> Dict[str, Any]:
    """导出当前状态的轻量级快照，适合序列化为 JSON 保存。

    注意: 不含 messages 和大型列表，仅保存关键控制字段。
    """
    import json

    al = state["al_state"]
    return {
        "session_id": state["session_id"],
        "pipeline_stage": state["pipeline_stage"],
        "iteration": al["iteration"],
        "rulebook_version": state["rulebook_version"],
        "screened_count": len(state["screened_records"]),
        "best_score": al["best_score_history"][-1] if al["best_score_history"] else None,
        "mlp_ready": state["mlp_ready"],
        "error_count": len(state["errors"]),
        "updated_at": state["updated_at"],
    }
