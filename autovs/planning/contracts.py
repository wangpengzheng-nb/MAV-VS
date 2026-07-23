"""Artifact Schema Registry and Action I/O Contract Registry.

Centralized, single-source-of-truth definitions for:
- What each artifact means and which formats it supports
- What each ActionType consumes and produces
- Which actions can produce a given artifact

Used by ToolUsePlannerAgent and WorkflowGraphBuilder.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from autovs.schemas import ActionType, StrictModel
from autovs.dag import (
    SCREENING_LIBRARY, NORMALIZED_LIBRARY, TARGET_STRUCTURE,
    POCKET_RESOLUTION, POCKET_CENTER, POCKET_SIZE,
    PREPARED_LIBRARY, STANDARDIZED_LIBRARY, IONIZED_LIBRARY,
    ENUMERATED_3D_SDF, LIGAND_PDBQT, CONVERTED_FORMAT,
    MOLECULE_PREP_REPORTS, MANIFEST_CSV, RECEPTOR_PDB, RECEPTOR_PDBQT,
    DOCKED_POSES, SCORES_CSV, SELECTED_POSES, COMPLEX_INDEX,
    PLIP_SCORES, TOP_HITS, HIT_COUNT,
)


# ═══════════════════════════════════════════════════════════════════════
# Artifact Schema Registry
# ═══════════════════════════════════════════════════════════════════════

class ArtifactSchema(StrictModel):
    """描述一个标准 artifact 的语义、允许格式和来源。"""
    artifact_key: str = Field(min_length=1, max_length=60)
    description: str
    allowed_formats: list[str] = Field(default_factory=list)
    multiple: bool = False            # 是否可以有多个实例
    user_provided: bool = False       # 是否可由 InputManifest 直接提供
    sensitive_path: bool = True       # 路径是否应在 LLM 上下文中隐藏


# 注册表：所有标准 artifact 定义
ARTIFACT_REGISTRY: dict[str, ArtifactSchema] = {
    SCREENING_LIBRARY: ArtifactSchema(
        artifact_key=SCREENING_LIBRARY,
        description="原始输入筛选分子库（SMILES 或 CSV 格式）",
        allowed_formats=["strict_smi_v1", "smi", "csv"],
        user_provided=True,
    ),
    NORMALIZED_LIBRARY: ArtifactSchema(
        artifact_key=NORMALIZED_LIBRARY,
        description="经验证和归一化后的分子库（去盐、标准化 SMILES）",
        allowed_formats=["strict_smi_v1", "smi"],
    ),
    TARGET_STRUCTURE: ArtifactSchema(
        artifact_key=TARGET_STRUCTURE,
        description="靶蛋白 3D 结构（PDB 或 mmCIF 格式）",
        allowed_formats=["PDB", "mmCIF", "pdb", "cif"],
        user_provided=True,
    ),
    POCKET_RESOLUTION: ArtifactSchema(
        artifact_key=POCKET_RESOLUTION,
        description="口袋定义结果（JSON，含中心、尺寸、残基列表）",
        allowed_formats=["JSON"],
    ),
    POCKET_CENTER: ArtifactSchema(
        artifact_key=POCKET_CENTER,
        description="口袋中心坐标 (x, y, z) 的三元组",
        allowed_formats=["tuple"],
    ),
    POCKET_SIZE: ArtifactSchema(
        artifact_key=POCKET_SIZE,
        description="口袋尺寸 (x, y, z) 的三元组",
        allowed_formats=["tuple"],
    ),
    PREPARED_LIBRARY: ArtifactSchema(
        artifact_key=PREPARED_LIBRARY,
        description="准备完成的配体库（SDF 格式，含 3D 坐标）",
        allowed_formats=["SDF", "sdf"],
    ),
    STANDARDIZED_LIBRARY: ArtifactSchema(
        artifact_key=STANDARDIZED_LIBRARY,
        description="ChEMBL 标准化后的 strict SMI 分子库",
        allowed_formats=["strict_smi_v1", "smi"],
    ),
    IONIZED_LIBRARY: ArtifactSchema(
        artifact_key=IONIZED_LIBRARY,
        description="Dimorphite-DL pH 枚举后的 strict SMI 分子库",
        allowed_formats=["strict_smi_v1", "smi"],
    ),
    ENUMERATED_3D_SDF: ArtifactSchema(
        artifact_key=ENUMERATED_3D_SDF,
        description="Gypsum-DL/RDKit 生成的 3D SDF 配体库",
        allowed_formats=["SDF", "sdf"],
    ),
    LIGAND_PDBQT: ArtifactSchema(
        artifact_key=LIGAND_PDBQT,
        description="Meeko 生成的配体 PDBQT 文件，仅供 Vina/PDBQT 路线使用",
        allowed_formats=["PDBQT", "pdbqt"],
    ),
    CONVERTED_FORMAT: ArtifactSchema(
        artifact_key=CONVERTED_FORMAT,
        description="Open Babel 转换得到的分子格式文件",
        allowed_formats=["SMI", "SDF", "PDB", "MOL2", "PDBQT"],
    ),
    MOLECULE_PREP_REPORTS: ArtifactSchema(
        artifact_key=MOLECULE_PREP_REPORTS,
        description="分子准备工具产生的诊断报告集合",
        allowed_formats=["JSON", "json", "list"],
        multiple=True,
    ),
    MANIFEST_CSV: ArtifactSchema(
        artifact_key=MANIFEST_CSV,
        description="分子清单 CSV（source_id, SMILES, 物化性质等）",
        allowed_formats=["CSV", "csv"],
    ),
    RECEPTOR_PDB: ArtifactSchema(
        artifact_key=RECEPTOR_PDB,
        description="清理后的受体 PDB（去水、去异质、仅蛋白原子）",
        allowed_formats=["PDB", "pdb"],
    ),
    RECEPTOR_PDBQT: ArtifactSchema(
        artifact_key=RECEPTOR_PDBQT,
        description="受体 PDBQT 文件（AutoDock 格式，含 Gasteiger 电荷）",
        allowed_formats=["PDBQT", "pdbqt"],
    ),
    DOCKED_POSES: ArtifactSchema(
        artifact_key=DOCKED_POSES,
        description="分子对接输出姿态（SDF，含对接打分）",
        allowed_formats=["SDF", "sdf"],
    ),
    SCORES_CSV: ArtifactSchema(
        artifact_key=SCORES_CSV,
        description="对接打分 CSV（source_id, docking_affinity, ...）",
        allowed_formats=["CSV", "csv"],
    ),
    SELECTED_POSES: ArtifactSchema(
        artifact_key=SELECTED_POSES,
        description="提取的最佳姿态（SDF，每个分子一个姿态）",
        allowed_formats=["SDF", "sdf"],
    ),
    COMPLEX_INDEX: ArtifactSchema(
        artifact_key=COMPLEX_INDEX,
        description="蛋白-配体复合物索引 JSON",
        allowed_formats=["JSON", "json"],
    ),
    PLIP_SCORES: ArtifactSchema(
        artifact_key=PLIP_SCORES,
        description="PLIP 相互作用指纹打分 CSV",
        allowed_formats=["CSV", "csv"],
    ),
    TOP_HITS: ArtifactSchema(
        artifact_key=TOP_HITS,
        description="最终排名 top-N 分子 CSV",
        allowed_formats=["CSV", "csv"],
    ),
    HIT_COUNT: ArtifactSchema(
        artifact_key=HIT_COUNT,
        description="命中分子数量",
        allowed_formats=["int"],
        sensitive_path=False,
    ),
}

for _ak, _as in ARTIFACT_REGISTRY.items():
    if _as.artifact_key != _ak:
        raise ValueError(f"Artifact registry key mismatch: {_ak} vs {_as.artifact_key}")


def get_artifact(key: str) -> ArtifactSchema | None:
    return ARTIFACT_REGISTRY.get(key)


# ═══════════════════════════════════════════════════════════════════════
# Action I/O Contract Registry
# ═══════════════════════════════════════════════════════════════════════

class ActionIOContract(StrictModel):
    """描述单个 ActionType 的输入/输出合约。"""
    action_type: ActionType
    scientific_role: str = ""
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    allowed_parameters: dict[str, str] = Field(default_factory=dict)
    default_quality_gates: list[str] = Field(default_factory=list)
    # 执行特征
    gpu_required: bool = False
    executor_type: str = "python"  # python, conda, slurm, apptainer
    # 是否可被 Planner 自动插入（服务拥有的步骤）
    service_owned: bool = False
    # 是否为 source producer（从无到有创建 artifact）
    # 区别于 transform producer（需要同名输入）
    is_source_producer: bool = False


ACTION_CONTRACTS: dict[ActionType, ActionIOContract] = {
    ActionType.INPUT_VALIDATION: ActionIOContract(
        action_type=ActionType.INPUT_VALIDATION,
        scientific_role="验证输入 PDB 和分子库格式，归一化分子库",
        required_inputs=[SCREENING_LIBRARY],
        optional_inputs=[TARGET_STRUCTURE],
        outputs=[NORMALIZED_LIBRARY],
        service_owned=True,
        is_source_producer=True,
    ),
    ActionType.TARGET_STRUCTURE_ACQUISITION: ActionIOContract(
        action_type=ActionType.TARGET_STRUCTURE_ACQUISITION,
        scientific_role="从 RCSB 下载经验证的实验共晶结构",
        required_inputs=[],
        optional_inputs=[],
        outputs=[TARGET_STRUCTURE],
        service_owned=True,
        is_source_producer=True,
    ),
    ActionType.POCKET_DEFINITION: ActionIOContract(
        action_type=ActionType.POCKET_DEFINITION,
        scientific_role="确定结合口袋的中心、尺寸和关键残基",
        required_inputs=[TARGET_STRUCTURE],
        optional_inputs=[],
        outputs=[POCKET_CENTER, POCKET_SIZE, POCKET_RESOLUTION],
        service_owned=True,
    ),
    ActionType.PROTEIN_PREPARATION: ActionIOContract(
        action_type=ActionType.PROTEIN_PREPARATION,
        scientific_role="准备受体蛋白：去水、加氢、生成 PDBQT",
        required_inputs=[TARGET_STRUCTURE],
        optional_inputs=[],
        outputs=[RECEPTOR_PDB, RECEPTOR_PDBQT],
        executor_type="conda",
    ),
    ActionType.MOLECULE_STANDARDIZATION: ActionIOContract(
        action_type=ActionType.MOLECULE_STANDARDIZATION,
        scientific_role="RDKit 标准化分子库",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[PREPARED_LIBRARY, MANIFEST_CSV],
    ),
    ActionType.MOLECULE_STANDARDIZATION_V2: ActionIOContract(
        action_type=ActionType.MOLECULE_STANDARDIZATION_V2,
        scientific_role="ChEMBL 标准化 + 去盐",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[STANDARDIZED_LIBRARY, NORMALIZED_LIBRARY, MOLECULE_PREP_REPORTS],
    ),
    ActionType.IONIZATION_ENUMERATION: ActionIOContract(
        action_type=ActionType.IONIZATION_ENUMERATION,
        scientific_role="pH 依赖离子化状态枚举",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[IONIZED_LIBRARY, NORMALIZED_LIBRARY, MOLECULE_PREP_REPORTS],
    ),
    ActionType.LIGAND_3D_ENUMERATION: ActionIOContract(
        action_type=ActionType.LIGAND_3D_ENUMERATION,
        scientific_role="3D-ready 配体枚举（质子化+互变+立体+构象）",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[ENUMERATED_3D_SDF, PREPARED_LIBRARY, MOLECULE_PREP_REPORTS],
    ),
    ActionType.PDBQT_PARAMETERIZATION: ActionIOContract(
        action_type=ActionType.PDBQT_PARAMETERIZATION,
        scientific_role="生成配体 PDBQT 文件（Gasteiger 电荷 + AutoDock 原子类型）",
        required_inputs=[PREPARED_LIBRARY],
        optional_inputs=[],
        outputs=[LIGAND_PDBQT, MOLECULE_PREP_REPORTS],
    ),
    ActionType.MOLECULAR_DOCKING: ActionIOContract(
        action_type=ActionType.MOLECULAR_DOCKING,
        scientific_role="分子对接（smina CPU / GNINA GPU）",
        required_inputs=[RECEPTOR_PDBQT, PREPARED_LIBRARY, POCKET_CENTER, POCKET_SIZE],
        optional_inputs=[MANIFEST_CSV],
        outputs=[DOCKED_POSES, SCORES_CSV],
        executor_type="slurm",
    ),
    ActionType.POSE_EXTRACTION: ActionIOContract(
        action_type=ActionType.POSE_EXTRACTION,
        scientific_role="提取每个分子的最佳对接姿态",
        required_inputs=[RECEPTOR_PDB, DOCKED_POSES],
        optional_inputs=[],
        outputs=[SELECTED_POSES, COMPLEX_INDEX],
    ),
    ActionType.INTERACTION_ANALYSIS: ActionIOContract(
        action_type=ActionType.INTERACTION_ANALYSIS,
        scientific_role="PLIP 蛋白-配体相互作用指纹分析",
        required_inputs=[COMPLEX_INDEX],
        optional_inputs=[],
        outputs=[PLIP_SCORES],
        executor_type="conda",
    ),
    ActionType.FINAL_RANKING: ActionIOContract(
        action_type=ActionType.FINAL_RANKING,
        scientific_role="综合证据归一化排序",
        required_inputs=[SCORES_CSV],
        optional_inputs=[],
        outputs=[TOP_HITS, HIT_COUNT],
    ),
    ActionType.REPORT_GENERATION: ActionIOContract(
        action_type=ActionType.REPORT_GENERATION,
        scientific_role="生成可复现虚拟筛选报告",
        required_inputs=[TOP_HITS],
        optional_inputs=[],
        outputs=[],
        service_owned=True,
    ),
    ActionType.STRUCTURE_ANALYSIS: ActionIOContract(
        action_type=ActionType.STRUCTURE_ANALYSIS,
        scientific_role="验证 PDB/mmCIF、检测配体、提取链信息",
        required_inputs=[TARGET_STRUCTURE],
        optional_inputs=[],
        outputs=[],
    ),
    ActionType.PROTEIN_REPAIR: ActionIOContract(
        action_type=ActionType.PROTEIN_REPAIR,
        scientific_role="补缺失原子/氢、替换非标准残基",
        required_inputs=[TARGET_STRUCTURE],
        optional_inputs=[],
        outputs=[TARGET_STRUCTURE],
    ),
    ActionType.PROTONATION: ActionIOContract(
        action_type=ActionType.PROTONATION,
        scientific_role="pH 依赖性质子化和电荷分配（PDB2PQR + PROPKA）",
        required_inputs=[TARGET_STRUCTURE],
        optional_inputs=[],
        outputs=[TARGET_STRUCTURE],
        executor_type="conda",
    ),
    ActionType.FORMAT_CONVERSION: ActionIOContract(
        action_type=ActionType.FORMAT_CONVERSION,
        scientific_role="分子格式转换（SMI↔SDF↔PDB↔MOL2）",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[CONVERTED_FORMAT, PREPARED_LIBRARY, MOLECULE_PREP_REPORTS],
    ),
    # ── 未实现处理器但已注册的 action ──
    ActionType.TARGET_STRUCTURE_PREDICTION: ActionIOContract(
        action_type=ActionType.TARGET_STRUCTURE_PREDICTION,
        scientific_role="AlphaFold/Boltz 结构预测",
        required_inputs=[],
        optional_inputs=[],
        outputs=[TARGET_STRUCTURE],
        gpu_required=True,
    ),
    ActionType.CONFORMER_GENERATION: ActionIOContract(
        action_type=ActionType.CONFORMER_GENERATION,
        scientific_role="RDKit ETKDGv3 3D 构象生成",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[PREPARED_LIBRARY],
    ),
    ActionType.PHYSICOCHEMICAL_FILTERING: ActionIOContract(
        action_type=ActionType.PHYSICOCHEMICAL_FILTERING,
        scientific_role="理化性质过滤（MW, LogP, PAINS, 反应基团）",
        required_inputs=[NORMALIZED_LIBRARY],
        optional_inputs=[],
        outputs=[NORMALIZED_LIBRARY],
    ),
    ActionType.ADMET_FILTERING: ActionIOContract(
        action_type=ActionType.ADMET_FILTERING,
        scientific_role="ADMET 风险预测",
        required_inputs=[SCORES_CSV],
        optional_inputs=[],
        outputs=[SCORES_CSV],
    ),
    ActionType.SHORT_MD: ActionIOContract(
        action_type=ActionType.SHORT_MD,
        scientific_role="10 ns GROMACS 短 MD 稳定性检查",
        required_inputs=[RECEPTOR_PDB, SELECTED_POSES],
        optional_inputs=[],
        outputs=[SCORES_CSV],
        gpu_required=True,
        executor_type="apptainer",
    ),
    ActionType.MOLECULAR_DYNAMICS: ActionIOContract(
        action_type=ActionType.MOLECULAR_DYNAMICS,
        scientific_role="100 ns GROMACS 生产 MD + MMGBSA",
        required_inputs=[RECEPTOR_PDB, SELECTED_POSES],
        optional_inputs=[],
        outputs=[SCORES_CSV],
        gpu_required=True,
        executor_type="apptainer",
    ),
    ActionType.DIVERSITY_SELECTION: ActionIOContract(
        action_type=ActionType.DIVERSITY_SELECTION,
        scientific_role="Murcko 骨架多样性筛选",
        required_inputs=[SCORES_CSV],
        optional_inputs=[],
        outputs=[SCORES_CSV],
    ),
}


def get_contract(action: ActionType) -> ActionIOContract | None:
    return ACTION_CONTRACTS.get(action)


def find_producers(artifact_key: str) -> list[ActionType]:
    """返回所有能产生指定 artifact 的 ActionType。"""
    return [
        action for action, contract in ACTION_CONTRACTS.items()
        if artifact_key in contract.outputs
    ]


def find_consumers(artifact_key: str) -> list[ActionType]:
    """返回所有需要消费指定 artifact 的 ActionType。"""
    return [
        action for action, contract in ACTION_CONTRACTS.items()
        if artifact_key in contract.required_inputs
        or artifact_key in contract.optional_inputs
    ]
