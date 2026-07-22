from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.agents.target_research.models import TargetIdentity, TargetIntent


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ActionType(str, Enum):
    INPUT_VALIDATION = "input_validation"
    TARGET_STRUCTURE_ACQUISITION = "target_structure_acquisition"
    TARGET_STRUCTURE_PREDICTION = "target_structure_prediction"
    PROTEIN_PREPARATION = "protein_preparation"
    POCKET_DEFINITION = "pocket_definition"
    MOLECULE_STANDARDIZATION = "molecule_standardization"
    CONFORMER_GENERATION = "conformer_generation"
    PHYSICOCHEMICAL_FILTERING = "physicochemical_filtering"
    DIVERSITY_SELECTION = "diversity_selection"
    MOLECULAR_DOCKING = "molecular_docking"
    POSE_EXTRACTION = "pose_extraction"
    INTERACTION_ANALYSIS = "interaction_analysis"
    ADMET_FILTERING = "admet_filtering"
    SHORT_MD = "short_md"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    FINAL_RANKING = "final_ranking"
    REPORT_GENERATION = "report_generation"
    STRUCTURE_ANALYSIS = "structure_analysis"


class ArtifactRef(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    format: str = Field(min_length=1, max_length=20)
    path: str | None = None


class ResourceProfile(StrictModel):
    executor: Literal["python", "conda", "slurm", "apptainer"] = "python"
    environment: str | None = None
    cpus: int = Field(default=1, ge=1, le=128)
    memory_gb: int = Field(default=4, ge=1, le=1024)
    gpu_required: bool = False
    timeout_seconds: int = Field(default=3600, ge=1)


class WorkflowStep(StrictModel):
    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    action_type: ActionType
    requires: list[str] = Field(default_factory=list)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    quality_gates: list[str] = Field(default_factory=list)
    resource_profile: ResourceProfile = Field(default_factory=ResourceProfile)

    @field_validator("requires")
    @classmethod
    def unique_requires(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("requires contains duplicates")
        return value


class WorkflowPlan(StrictModel):
    plan_version: Literal["1.0"] = "1.0"
    strategy_id: str = Field(min_length=1, max_length=120)
    steps: list[WorkflowStep] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dag(self) -> "WorkflowPlan":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique")
        seen: set[str] = set()
        for step in self.steps:
            missing = set(step.requires) - seen
            if missing:
                raise ValueError(f"{step.step_id} depends on unknown or later steps: {sorted(missing)}")
            seen.add(step.step_id)
        return self


class PocketSpec(StrictModel):
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] = (24.0, 24.0, 24.0)
    key_residues: list[str] = Field(default_factory=list)
    cocrystal_ligand: str | None = None

    @field_validator("size")
    @classmethod
    def valid_box_size(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(axis < 8.0 or axis > 60.0 for axis in value):
            raise ValueError("pocket box dimensions must each be between 8 and 60 Angstrom")
        return value


class PocketSource(str, Enum):
    USER_COORDINATES = "user_coordinates"
    COCRYSTAL_LIGAND = "cocrystal_ligand"
    VERIFIED_RESEARCH_STRUCTURE = "verified_research_structure"
    KEY_RESIDUES = "key_residues"


class PocketConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PocketEvidence(StrictModel):
    kind: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1000)
    value: float | str | bool | None = None


class PocketQualityGate(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    status: Literal["passed", "degraded", "failed", "not_run"]
    detail: str = Field(default="", max_length=1000)


class PocketCandidate(StrictModel):
    pocket_id: str = Field(pattern=r"^pocket_[a-f0-9]{12}$")
    rank: int = Field(ge=1)
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    source: PocketSource
    confidence: PocketConfidence
    chain_ids: list[str] = Field(default_factory=list)
    residues: list[str] = Field(default_factory=list)
    evidence: list[PocketEvidence] = Field(default_factory=list)
    quality_gates: list[PocketQualityGate] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)


class PocketResolution(StrictModel):
    resolution_version: Literal["1.0"] = "1.0"
    protein_path: str
    selected_pocket: PocketCandidate
    alternate_pockets: list[PocketCandidate] = Field(default_factory=list, max_length=2)
    research_pdb_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def selected_must_be_usable(self) -> "PocketResolution":
        if self.selected_pocket.confidence == PocketConfidence.LOW:
            raise ValueError("selected pocket cannot have low confidence")
        ids = [self.selected_pocket.pocket_id, *(item.pocket_id for item in self.alternate_pockets)]
        if len(ids) != len(set(ids)):
            raise ValueError("pocket ids must be unique")
        return self


class TaskRequest(StrictModel):
    query: str = Field(min_length=10, max_length=5000)
    protein_path: str | None = None
    library_path: str | None = None
    protein_original_name: str | None = None
    library_original_name: str | None = None
    input_manifest_path: str | None = None
    pocket: PocketSpec = Field(default_factory=PocketSpec)
    known_actives_path: str | None = None
    ph: float = Field(default=7.4, ge=0.0, le=14.0)
    cpu_only: bool = False
    resume: bool = True
    target_identity: TargetIdentity | None = None
    screening_intent: TargetIntent | None = None


class LibraryAsset(StrictModel):
    source: Literal["user", "builtin"]
    locked: Literal[True] = True
    format: Literal["strict_smi_v1"] = "strict_smi_v1"
    path: str
    normalized_path: str | None = None
    version: str | None = None
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_filename: str | None = None
    total_records: int | None = Field(default=None, ge=0)
    accepted_records: int | None = Field(default=None, ge=0)
    quarantined_records: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_counts_and_version(self) -> "LibraryAsset":
        if self.source == "builtin" and not self.version:
            raise ValueError("builtin library requires a version")
        counts = (self.total_records, self.accepted_records, self.quarantined_records)
        if all(value is not None for value in counts) and self.accepted_records + self.quarantined_records != self.total_records:  # type: ignore[operator]
            raise ValueError("accepted_records + quarantined_records must equal total_records")
        return self


class TargetAsset(StrictModel):
    source: Literal["user", "research"]
    locked: bool
    path: str | None = None
    pdb_id: str | None = Field(default=None, pattern=r"^[0-9][A-Za-z0-9]{3}$")
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    original_filename: str | None = None

    @model_validator(mode="after")
    def validate_lock(self) -> "TargetAsset":
        if self.source == "user" and not self.locked:
            raise ValueError("uploaded target asset must be locked")
        if self.locked and (not self.path or not self.sha256):
            raise ValueError("locked target asset requires path and sha256")
        return self


class InputManifest(StrictModel):
    manifest_version: Literal["1.0"] = "1.0"
    query: str = Field(min_length=10, max_length=5000)
    library_asset: LibraryAsset
    target_asset: TargetAsset
    expert_pocket: PocketSpec = Field(default_factory=PocketSpec)
    warnings: list[str] = Field(default_factory=list)
    constraint_summary: list[str] = Field(default_factory=list)
    target_identity: TargetIdentity | None = None
    screening_intent: TargetIntent | None = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobRecord(StrictModel):
    job_id: str
    task_id: str
    step_id: str
    action_type: ActionType
    status: JobStatus
    attempt: int = 0
    slurm_job_id: str | None = None
    message: str = ""
    created_at: str
    updated_at: str


class ToolCapability(StrictModel):
    action_type: ActionType
    name: str
    description: str
    availability: Literal["available", "degraded", "unavailable"]
    executor: Literal["python", "conda", "slurm", "apptainer"]
    input_formats: list[str]
    output_formats: list[str]
    gpu_required: bool = False
    reason: str = ""


class ExecutorType(str, Enum):
    SUBPROCESS = "subprocess"          # 直接调用外部二进制
    PYTHON_MODULE = "python_module"    # conda run -n <env> python -m <module>
    APPTAINER = "apptainer"            # Apptainer/Singularity 容器
    SLURM = "slurm"                    # Slurm 作业提交


class ExecutorConfig(StrictModel):
    """结构化工具执行器注册表。"""
    name: str = Field(min_length=1, max_length=40)
    executor: ExecutorType
    path: str | None = None             # 二进制或脚本路径（subprocess / apptainer）
    env: str | None = None              # conda 环境名（python_module）
    module: str | None = None           # Python 模块名（python_module）
    env_hint: str = ""                  # 仅文档：该二进制来自哪个 conda 环境
    health_check: str = ""              # 健康检查命令或模块导入名
    description: str = ""
    gpu_required: bool = False

    def resolved_path(self, project_root: str = "") -> Path | None:
        if not self.path:
            return None
        from pathlib import Path as _Path
        p = _Path(self.path)
        if p.is_absolute():
            return p
        return (_Path(project_root) / p).resolve() if project_root else p

    def exists(self, project_root: str = "") -> bool:
        p = self.resolved_path(project_root)
        return p is not None and p.is_file()


class MoleculeResult(StrictModel):
    source_id: str
    structure_id: str | None = None
    smiles: str
    docking_affinity: float | None = None
    cnn_score: float | None = None
    cnn_affinity: float | None = None
    cnn_vs: float | None = None
    plip_score: float | None = None
    admet_risk: float | None = None
    scaffold: str = ""
    short_md_stable: bool | None = None
    mmgbsa_delta_total: float | None = None
    final_score: float | None = None
    rank: int | None = None
    status: JobStatus = JobStatus.SUCCEEDED
    notes: list[str] = Field(default_factory=list)


def ensure_existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    return path


def parse_gene_from_query(query: str, gene: str, log: list[str]) -> str:
    user_gene = ""
    m = re.match(r'靶点基因:\s*([A-Za-z0-9][-A-Za-z0-9]*)', query)
    if m:
        user_gene = m.group(1).upper()
        llm_gene = gene.upper()
        if user_gene and user_gene != llm_gene:
            # 用户输入是 LLM 结果的子串（如 EP2 vs PTGER2）→ 保留 LLM 的完整名
            if user_gene in llm_gene or llm_gene in user_gene:
                log.append(f"📌 用户输入别名: {user_gene} → LLM映射: {gene}")
            else:
                # 完全不同 → 以用户输入为准
                log.append(f"📌 用户指定基因: {user_gene}（覆盖LLM解析: {gene}）")
                gene = user_gene
        elif user_gene:
            gene = user_gene
    return gene
