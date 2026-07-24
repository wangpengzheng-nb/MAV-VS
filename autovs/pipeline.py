from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from autovs.capabilities import list_capabilities
from autovs.compiler import choose_executable_strategy, compile_strategy
from autovs.config import Settings, load_settings
from autovs.db import StateStore
from autovs.reporting import generate_failure_report
from autovs.dag import (
    NORMALIZED_LIBRARY, POCKET_RESOLUTION as POCKET_RESOLUTION_KEY,
    SCREENING_LIBRARY, TARGET_STRUCTURE, TaskPaused as DAGTaskPaused,
    execute_workflow_plan,
)
from autovs.library import (
    migrate_legacy_library, validate_smi_structure,
    verify_default_library,
)
from autovs.schemas import (
    ActionType, InputManifest, JobStatus, LibraryAsset, TargetAsset,
    TaskRequest, WorkflowPlan, WorkflowStep,
)
from autovs.security import sha256_file
from autovs.tool_manager import ToolManager
from src.agents.target_research.models import TargetIdentity, TargetIntent


class _TaskPaused(Exception):
    """内部信号：任务已暂停，用于从 _run_existing 中优雅退出。"""
    pass


PIPELINE_PHASES = [
    ("input_validation", "输入校验"),
    ("target_research", "靶点调研"),
    ("strategy_generation", "策略生成"),
    ("strategy_voting", "全排列投票"),
    ("strategy_evolution", "策略进化"),
    ("strategy_selection", "可执行策略选择"),
    ("target_structure_acquisition", "靶结构获取"),
    ("pocket_definition", "口袋确定"),
    ("molecule_standardization", "分子准备"),
    ("protein_preparation", "蛋白准备"),
    ("molecular_docking", "分子对接"),
    ("pose_extraction", "姿态提取"),
    ("interaction_analysis", "PLIP 相互作用"),
    ("final_ranking", "候选排序"),
    ("report_generation", "报告生成"),
]

ACTION_PHASE = {
    ActionType.INPUT_VALIDATION: "input_validation",
    ActionType.TARGET_STRUCTURE_ACQUISITION: "target_structure_acquisition",
    ActionType.TARGET_STRUCTURE_PREDICTION: "target_structure_acquisition",
    ActionType.POCKET_DEFINITION: "pocket_definition",
    ActionType.MOLECULE_STANDARDIZATION: "molecule_standardization",
    ActionType.CONFORMER_GENERATION: "molecule_standardization",
    ActionType.PHYSICOCHEMICAL_FILTERING: "molecule_standardization",
    ActionType.PROTEIN_PREPARATION: "protein_preparation",
    ActionType.MOLECULAR_DOCKING: "molecular_docking",
    ActionType.POSE_EXTRACTION: "pose_extraction",
    ActionType.INTERACTION_ANALYSIS: "interaction_analysis",
    ActionType.FINAL_RANKING: "final_ranking",
    ActionType.STRUCTURE_ANALYSIS: "target_structure_acquisition",
    ActionType.PROTEIN_REPAIR: "protein_preparation",
    ActionType.PROTONATION: "protein_preparation",
    ActionType.MOLECULE_STANDARDIZATION_V2: "molecule_standardization",
    ActionType.LIGAND_3D_ENUMERATION: "molecule_standardization",
    ActionType.IONIZATION_ENUMERATION: "molecule_standardization",
    ActionType.PDBQT_PARAMETERIZATION: "molecule_standardization",
    ActionType.FORMAT_CONVERSION: "molecule_standardization",
    ActionType.POSE_VALIDATION: "pose_extraction",
    ActionType.POCKET_PREDICTION: "pocket_definition",
    ActionType.DIFFDOCK_DOCKING: "molecular_docking",
    ActionType.GEOMETRIC_POCKET_DETECTION: "pocket_definition",
    ActionType.PHARMACOPHORE_SCREENING: "molecule_standardization",
    ActionType.STRUCTURAL_HOMOLOGY_SEARCH: "target_research",
}

LLM_ONLY_PHASES = ("target_research", "strategy_generation", "strategy_voting", "strategy_evolution")


class PipelineService:
    """The single application service used by CLI, Web, and tests."""

    def __init__(self, settings: Settings | None = None, *, enable_slurm_poller: bool = True):
        self.settings = settings or load_settings()
        self.store = StateStore(self.settings.database_path)
        self.tools = ToolManager(self.settings, self.store)
        self._cancel_events: dict[str, threading.Event] = {}
        self._pause_events: dict[str, threading.Event] = {}
        # Slurm 自动轮询器: GPU 作业完成后自动恢复 pipeline
        self._slurm_poller = None
        if enable_slurm_poller:
            from autovs.slurm_poller import SlurmPoller
            self._slurm_poller = SlurmPoller(self, interval=30)
            self._slurm_poller.start()

    def submit(self, request: TaskRequest, *, use_llm_planning: bool = True) -> str:
        staged, task_dir = self._stage_request(request)
        task_id = self.store.create_task(staged.model_dump(mode="json"), task_dir)
        self._initialize_progress(task_id, use_llm_planning)
        self._cancel_events[task_id] = threading.Event()
        threading.Thread(target=self._run_existing, args=(task_id, use_llm_planning), daemon=True).start()
        return task_id

    def run_sync(self, request: TaskRequest, *, use_llm_planning: bool = True) -> dict:
        staged, task_dir = self._stage_request(request)
        task_id = self.store.create_task(staged.model_dump(mode="json"), task_dir)
        self._initialize_progress(task_id, use_llm_planning)
        self._run_existing(task_id, use_llm_planning)
        return self.get_task(task_id) or {}

    def resume(self, task_id: str) -> None:
        task = self.store.get_task(task_id)
        if not task:
            raise ValueError(f"unknown task_id: {task_id}")
        allowed = {JobStatus.FAILED.value, JobStatus.CANCELLED.value, JobStatus.PAUSED.value}
        if task["status"] not in allowed:
            raise ValueError(f"只能续跑已暂停、已取消或已失败的任务，当前状态: {task['status']}")
        self._pause_events.pop(task_id, None)
        self._cancel_events.pop(task_id, None)
        threading.Thread(target=self._run_existing, args=(task_id, True), daemon=True).start()

    def get_task(self, task_id: str) -> dict | None:
        task = self.store.get_task(task_id)
        if task:
            task["jobs"] = self.store.list_jobs(task_id)
            task["artifacts"] = self.store.list_artifacts(task_id)
            task["progress"] = self.store.list_progress(task_id)
            manifest_path = task.get("request", {}).get("input_manifest_path")
            if manifest_path and Path(manifest_path).is_file():
                task["input_manifest"] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return task

    def cancel_task(self, task_id: str) -> bool:
        """取消正在运行的任务。返回True表示已发出取消信号。"""
        task = self.store.get_task(task_id)
        if not task:
            return False
        if task["status"] not in {"pending", "running"}:
            return False
        # 发出取消信号
        cancel_evt = self._cancel_events.get(task_id)
        if cancel_evt:
            cancel_evt.set()
        self.store.update_task(task_id, JobStatus.CANCELLED)
        self.store.cancel_running_progress(task_id)
        return True

    def pause_task(self, task_id: str) -> bool:
        """暂停正在运行的任务，保留已完成阶段的成果。返回True表示已发出暂停信号。"""
        task = self.store.get_task(task_id)
        if not task:
            return False
        if task["status"] != JobStatus.RUNNING.value:
            return False
        pause_evt = self._pause_events.get(task_id)
        if pause_evt:
            pause_evt.set()
        return True

    def _is_paused(self, task_id: str) -> bool:
        evt = self._pause_events.get(task_id)
        return evt.is_set() if evt else False

    def _is_cancelled(self, task_id: str) -> bool:
        evt = self._cancel_events.get(task_id)
        return evt.is_set() if evt else False

    def delete_task(self, task_id: str) -> dict:
        """永久删除已完成/失败/已取消的任务（数据库记录 + 磁盘目录）。
        对运行中或等待中的任务会拒绝删除。"""
        import shutil
        task = self.store.get_task(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")
        if task["status"] in {JobStatus.PENDING.value, JobStatus.RUNNING.value}:
            raise ValueError(f"任务正在运行或等待中，无法删除。请先取消任务。")
        # 清理取消事件引用
        self._cancel_events.pop(task_id, None)
        # 删除磁盘上的任务目录
        task_dir = Path(task.get("task_dir", ""))
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
        # 删除数据库记录
        self.store.delete_task(task_id)
        return {"task_id": task_id, "deleted": True}

    def _initialize_progress(self, task_id: str, use_llm_planning: bool) -> None:
        self.store.initialize_progress(task_id, PIPELINE_PHASES)
        if not use_llm_planning:
            for phase_id in LLM_ONLY_PHASES:
                self.store.update_progress(
                    task_id, phase_id, JobStatus.SKIPPED,
                    message="基础链路诊断模式跳过 LLM 规划",
                )

    def _stage_request(self, request: TaskRequest) -> tuple[TaskRequest, Path]:
        source_protein = Path(request.protein_path).expanduser().resolve() if request.protein_path else None
        if source_protein and not source_protein.is_file():
            raise ValueError(f"protein_path does not exist or is not a file: {source_protein}")
        user_library = bool(request.library_path)
        source_library = (Path(request.library_path).expanduser().resolve()
                          if request.library_path else self.settings.default_library_path)
        if not source_library.is_file():
            raise ValueError(f"library_path does not exist or is not a file: {source_library}")
        validate_smi_structure(
            source_library, max_molecules=int(self.settings.limit("max_library_molecules", 1_000_000)),
        )
        if not user_library:
            cfg = self.settings.library()
            check = verify_default_library(source_library, str(cfg.get("sha256", "")), int(cfg.get("molecule_count", 0)))
            if check["status"] != "available":
                raise ValueError(str(check.get("reason", "default library is unavailable")))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        identity_key = request.target_identity.identity_fingerprint if request.target_identity else "unresolved"
        fingerprint = hashlib.sha256(
            f"{request.query}|{identity_key}|{source_protein}|{source_library}".encode()
        ).hexdigest()[:8]
        task_dir = self.settings.task_root / f"task_{stamp}_{fingerprint}"
        inputs = task_dir / "inputs"; inputs.mkdir(parents=True, exist_ok=False)
        protein = None
        if source_protein:
            protein = inputs / "target_structure.pdb"
            shutil.copy2(source_protein, protein)
        if user_library:
            library = inputs / f"screening_library{source_library.suffix.lower()}"
            shutil.copy2(source_library, library)
        else:
            library = source_library
        warnings = []
        if not user_library:
            warnings.append("未上传分子库：本任务强制使用内置 PocketXMol curated 87K 分子库。")
        if protein is None:
            warnings.append("未上传蛋白结构：系统将优先获取经过验证的实验共晶结构；若不存在则由策略阶段提出结构预测路线。")
        cfg = self.settings.library() if not user_library else {}
        manifest_path = task_dir / "input_manifest.json"
        manifest = InputManifest(
            query=request.query,
            library_asset=LibraryAsset(
                source="user" if user_library else "builtin", path=str(library), sha256=sha256_file(library),
                version=str(cfg.get("version")) if cfg else None,
                original_filename=request.library_original_name or (source_library.name if user_library else None),
            ),
            target_asset=TargetAsset(
                source="user" if protein else "research", locked=bool(protein), path=str(protein) if protein else None,
                sha256=sha256_file(protein) if protein else None,
                original_filename=request.protein_original_name or (source_protein.name if source_protein else None),
            ),
            expert_pocket=request.pocket, warnings=warnings,
            target_identity=request.target_identity,
            screening_intent=request.screening_intent,
            constraint_summary=[
                "screening_library is immutable and may not be replaced by an external library",
                "uploaded target_structure is immutable" if protein else "target_structure must come from verified RCSB holo candidates",
                "pocket coordinates are deterministic tool outputs, never LLM-authored",
            ],
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        staged = request.model_copy(update={
            "protein_path": str(protein) if protein else None, "library_path": str(library),
            "input_manifest_path": str(manifest_path),
        })
        (task_dir / "request.json").write_text(staged.model_dump_json(indent=2), encoding="utf-8")
        return staged, task_dir

    def _run_existing(self, task_id: str, use_llm_planning: bool) -> None:
        task = self.store.get_task(task_id)
        if not task:
            return
        self.store.update_task(task_id, JobStatus.RUNNING)
        # 重置 pause 信号（resume 时已清理，submit 时是新任务）
        self._pause_events.setdefault(task_id, threading.Event()).clear()
        task_dir = Path(task["task_dir"])
        request = TaskRequest.model_validate(task["request"])
        rejected: list[dict] = []
        selected_strategy: dict[str, Any] = {}
        run_warnings: list[str] = []

        # 读取当前进度，判断哪些阶段已完成（用于续跑跳过）
        def _phase_done(phase_id: str) -> bool:
            for p in self.store.list_progress(task_id):
                if p["phase_id"] == phase_id and p["status"] == "succeeded":
                    return True
            return False

        def _check_pause() -> None:
            """如果收到暂停信号，优雅退出。"""
            if self._is_paused(task_id):
                self.store.update_task(task_id, JobStatus.PAUSED)
                raise _TaskPaused()

        try:
            _check_pause()
            if self._is_cancelled(task_id):
                raise RuntimeError("task cancelled by user")
            if not request.input_manifest_path:
                request = self._migrate_legacy_request(task_id, request, task_dir)

            # ── 阶段1: 输入校验 ──
            if not _phase_done("input_validation"):
                validation = self._run_step(
                    task_id, WorkflowStep(step_id="input-validation", action_type=ActionType.INPUT_VALIDATION),
                    {"protein_path": request.protein_path, "library_path": request.library_path,
                     "input_manifest_path": request.input_manifest_path},
                )
                manifest = self._update_library_manifest(request, validation)
                normalized_library = str(validation["normalized_library"])
            else:
                manifest = self._read_input_manifest(request)
                normalized_library = manifest.library_asset.normalized_path or manifest.library_asset.path
                if not normalized_library:
                    raise ValueError("续跑失败: 找不到归一化分子库路径")

            _check_pause()

            # ── 阶段2: LLM 规划（内部各子阶段已支持 checkpoint 续跑）──
            if use_llm_planning:
                planning = self._run_planning(task_id, request, task_dir, manifest)
            else:
                if not request.protein_path:
                    raise ValueError("baseline mode requires an uploaded preprocessed PDB structure")
                plan = build_cpu_baseline_plan()
                planning = {"mode": "deterministic_cpu_baseline", "ranked_names": [plan.strategy_id]}

            _check_pause()

            readiness = planning.get("research", {}).get("structure_readiness", {})
            if (use_llm_planning and not request.protein_path
                    and readiness.get("predicted_structure_required")):
                prediction_cap = next(
                    (cap for cap in list_capabilities(self.settings)
                     if cap.action_type == ActionType.TARGET_STRUCTURE_PREDICTION),
                    None,
                )
                if prediction_cap and prediction_cap.availability != "unavailable":
                    run_warnings.append(
                        "未找到可用实验结构，将通过 target_structure_prediction 节点提交 AlphaFold3 任务"
                    )
                else:
                    reason = (
                        prediction_cap.reason if prediction_cap
                        else "AlphaFold/Boltz structure prediction adapter is not configured yet"
                    )
                    gap = {
                        "action_type": ActionType.TARGET_STRUCTURE_PREDICTION.value,
                        "availability": "unavailable",
                        "reason": reason,
                        "recommendations": readiness.get("acquisition_recommendations", []),
                    }
                    gap_path = task_dir / "structure_capability_gap.json"
                    gap_path.write_text(json.dumps(gap, ensure_ascii=False, indent=2), encoding="utf-8")
                    self._index_artifact(task_id, "structure_capability_gap", gap_path)
                    self.store.update_progress(
                        task_id, "strategy_selection", JobStatus.FAILED,
                        message="策略需要预测靶结构，但 AlphaFold3/Boltz 工具尚未可用",
                        error=gap["reason"], metadata=gap,
                    )
                    raise RuntimeError(
                        "capability gap: target_structure_prediction is required because no verified "
                        f"experimental holo structure is available; {reason}"
                    )
            # ── 阶段3: 策略选择 ──
            plan_path = task_dir / "workflow_plan.json"
            if _phase_done("strategy_selection") and plan_path.is_file():
                plan = WorkflowPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
                for candidate in planning.get("evolved_strategies", []):
                    if plan.strategy_id in {
                        str(candidate.get("strategy_id", "")),
                        str(candidate.get("strategy_name", "")),
                    }:
                        selected_strategy = candidate
                        break
            elif use_llm_planning:
                self.store.update_progress(task_id, "strategy_selection", JobStatus.RUNNING,
                                           message="按投票排名校验候选策略")
                selected_strategy, plan, rejected = choose_executable_strategy(
                    planning["ranked_names"], planning["evolved_strategies"], input_manifest=manifest,
                )
                self.store.update_progress(
                    task_id, "strategy_selection", JobStatus.SUCCEEDED,
                    message=f"已选择可执行策略：{plan.strategy_id}",
                    metadata={"strategy_id": plan.strategy_id, "rejected_count": len(rejected)},
                )
                plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                self._index_artifact(task_id, "workflow_plan", plan_path)
            else:
                if not request.protein_path:
                    raise ValueError("baseline mode requires an uploaded preprocessed PDB structure")
                plan = build_cpu_baseline_plan()
                plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                self._index_artifact(task_id, "workflow_plan", plan_path)
                planning = {"mode": "deterministic_cpu_baseline", "ranked_names": [plan.strategy_id]}
                selected_strategy = {
                    "strategy_id": plan.strategy_id,
                    "strategy_name": plan.strategy_id,
                    "pipeline": [step.model_dump(mode="json") for step in plan.steps],
                }
                self.store.update_progress(task_id, "strategy_selection", JobStatus.SUCCEEDED,
                                           message="已选择确定性 CPU 基线策略",
                                           metadata={"strategy_id": plan.strategy_id})

            _check_pause()

            # ── 阶段3.5: 工具使用规划 ──
            planner_mode = os.environ.get("AUTOVS_PLANNER_MODE", "tool_use")
            planning_result_path = task_dir / "tool_planning.json"
            planner_warnings: list[str] = []

            if planner_mode == "tool_use" and planning_result_path.is_file():
                plan = WorkflowPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
            elif planner_mode == "tool_use":
                # 仅在 ToolUsePlanner checkpoint 不存在时规划；续跑复用已有 plan。
                try:
                    from src.agents.tool_use_planner import ToolUsePlannerAgent
                    from autovs.planning import PlannerConstraints

                    planner = ToolUsePlannerAgent(
                        settings=self.settings, llm_client=None,
                    )
                    planner_result = planner.plan(
                        strategy=selected_strategy or (
                            planning.get("evolved_strategies", [{}])[0]
                            if planning.get("evolved_strategies") else {}
                        ),
                        input_manifest=manifest,
                        constraints=PlannerConstraints(cpu_only=getattr(request, "cpu_only", False)),
                    )
                    plan = planner_result.plan
                    planner_warnings = planner_result.warnings
                    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                    planning_result_path.write_text(
                        planner_result.model_dump_json(indent=2), encoding="utf-8",
                    )
                    self._index_artifact(task_id, "workflow_plan", plan_path)
                    self._index_artifact(task_id, "tool_planning", planning_result_path)

                    self.store.update_progress(
                        task_id, "strategy_selection", JobStatus.SUCCEEDED,
                        message=f"工具使用规划完成：{len(plan.steps)} 步骤，{len(planner_result.capability_gaps)} 能力缺口",
                        metadata={
                            "strategy_id": plan.strategy_id,
                            "step_count": len(plan.steps),
                            "capability_gaps": planner_result.capability_gaps,
                            "planner_warnings": planner_warnings,
                        },
                    )
                except Exception as exc:
                    planning_error_path = task_dir / "tool_planning_error.json"
                    planning_error_path.write_text(json.dumps({
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    self._index_artifact(task_id, "tool_planning_error", planning_error_path)
                    # 回退到 legacy compiler
                    run_warnings.append(f"ToolUsePlanner 失败，回退到 legacy compile_strategy: {exc}")
                    if plan_path.is_file():
                        plan = WorkflowPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
                    else:
                        raise RuntimeError(f"规划失败且无可回退计划: {exc}") from exc

            _check_pause()

            # ── 阶段4-5: DAG 工作流执行 ──
            # 构建 artifact_state，将所有步骤交给 DAG executor 按拓扑顺序执行。
            artifact_state: dict[str, Any] = {
                SCREENING_LIBRARY: request.library_path,
                NORMALIZED_LIBRARY: normalized_library,
                "_research_path": str(task_dir / "research.json") if (task_dir / "research.json").is_file() else "",
                "_selected_strategy_id": plan.strategy_id,
            }
            if request.protein_path:
                artifact_state[TARGET_STRUCTURE] = request.protein_path

            result = execute_workflow_plan(
                task_id, plan,
                tools=self.tools,
                artifact_state=artifact_state,
                store=self.store,
                task_dir=task_dir,
                request=request,
                planning={**planning, "warnings": run_warnings},
                rejected_strategies=rejected,
                update_progress=lambda phase_id, status, **kw: self.store.update_progress(
                    task_id, phase_id, status, **kw,
                ),
                is_paused=lambda: self._is_paused(task_id),
            )
            # Merge pocket resolution into planning metadata for downstream consumers.
            if POCKET_RESOLUTION_KEY in artifact_state:
                planning["pocket_resolution"] = artifact_state[POCKET_RESOLUTION_KEY]
            self.store.finish_pending_progress(task_id, message="未包含在本次可执行策略中")
            if result.get("status") == "failed":
                self.store.update_task(
                    task_id, JobStatus.FAILED, result=result,
                    error="workflow execution failed; see reports and progress for details",
                )
            else:
                self.store.update_task(task_id, JobStatus.SUCCEEDED, result=result)

        except Exception as exc:
            if self._is_paused(task_id) or isinstance(exc, (DAGTaskPaused, _TaskPaused)):
                # 暂停是正常行为，不记录错误；任务状态必须显式落库，供 Web 端续跑。
                self.store.update_task(task_id, JobStatus.PAUSED)
                return
            error = f"{type(exc).__name__}: {exc}"
            self.store.fail_running_progress(task_id, error=error)
            self.store.finish_pending_progress(task_id, message="上游阶段失败，未执行")
            diagnostic = task_dir / "pipeline_error.json"
            diagnostic.write_text(json.dumps({
                "task_id": task_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.add_artifact(task_id, None, "pipeline_error", diagnostic, "JSON", sha256_file(diagnostic))
            reports = generate_failure_report(task_id, task_dir, request=request.model_dump(mode="json"), error=error)
            for name, raw_path in reports.items():
                path = Path(raw_path)
                self.store.add_artifact(task_id, None, name, path, path.suffix.lstrip(".").upper(), sha256_file(path))
            failure = {"task_id": task_id, "status": "failed", "error": error, "reports": reports}
            (task_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.update_task(task_id, JobStatus.FAILED, result=failure, error=failure["error"])

    def _run_planning(self, task_id: str, request: TaskRequest, task_dir: Path,
                      manifest: InputManifest) -> dict:
        from src.agents.expert_committee import REVIEWER_CONFIGS, TournamentReviewer
        from src.agents.judge_agent import VoteAggregator
        from src.agents.strategy_evolver import StrategyEvolver
        from src.agents.strategy_generator import StrategyGeneratorAgent
        from src.agents.target_scout import TargetScoutAgent

        # 检查各子阶段是否已完成（支持从暂停点继续）
        def _sub_phase_done(phase_id: str) -> bool:
            for p in self.store.list_progress(task_id):
                if p["phase_id"] == phase_id and p["status"] == "succeeded":
                    return True
            return False

        # ── 子阶段1: 靶点调研 ──
        research_path = task_dir / "research.json"
        research = {}
        can_reuse = _sub_phase_done("target_research") and research_path.is_file()
        if can_reuse:
            research = json.loads(research_path.read_text(encoding="utf-8"))
            expected_identity = request.target_identity.identity_fingerprint if request.target_identity else ""
            actual_identity = (research.get("identity") or {}).get("identity_fingerprint", "")
            can_reuse = (
                research.get("research_version") == "2.0"
                and research.get("_user_query") == request.query
                and (not expected_identity or expected_identity == actual_identity)
            )
        if not can_reuse:
            self.store.update_progress(task_id, "target_research", JobStatus.RUNNING,
                                       message="正在解析自然语言、消歧靶点并检索可追溯证据")
            target_hint = (request.screening_intent.target_text if request.screening_intent else "")
            selected_accession = (request.target_identity.uniprot_accession if request.target_identity else "")
            raw_dir = task_dir / "research_raw"
            research = TargetScoutAgent(snapshot_dir=raw_dir).deep_research(
                request.query, fetch_structure_coordinates=False,
                target_hint=target_hint, selected_accession=selected_accession,
            )
            research["_user_query"] = request.query
            research_path.write_text(json.dumps(research, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            self._index_artifact(task_id, "research", research_path)
            for snapshot in sorted(raw_dir.glob("*.json")):
                self._index_artifact(task_id, f"research_raw_{snapshot.stem}", snapshot)
            self._lock_research_identity(task_id, request, manifest, research, task_dir)
            self.store.update_progress(task_id, "target_research", JobStatus.SUCCEEDED,
                                       message=f"靶点调研已完成：{research.get('gene_symbol')} / {research.get('target_uniprot_id')}",
                                       metadata={
                                           "research_status": research.get("status"),
                                           "input_fingerprint": research.get("input_fingerprint"),
                                           "identity_fingerprint": (research.get("identity") or {}).get("identity_fingerprint"),
                                           "evidence_gaps": research.get("evidence_gaps", []),
                                       })
        execution_context = self._execution_context(manifest)
        research["_execution_context"] = execution_context
        binding_rules = json.dumps(execution_context, ensure_ascii=False, separators=(",", ":"))
        if self._is_paused(task_id): self.store.update_task(task_id, JobStatus.PAUSED); raise _TaskPaused()

        # ── 子阶段2: 策略生成 ──
        if _sub_phase_done("strategy_generation"):
            strategies_path = task_dir / "strategies.json"
            strategies = json.loads(strategies_path.read_text(encoding="utf-8")) if strategies_path.is_file() else []
        else:
            self.store.update_progress(task_id, "strategy_generation", JobStatus.RUNNING,
                                       message="正在生成结构化虚拟筛选策略")
            strategies = StrategyGeneratorAgent().generate_strategies(research, prior_knowledge=binding_rules)["strategies"]
            strategies_path = task_dir / "strategies.json"
            strategies_path.write_text(json.dumps(strategies, ensure_ascii=False, indent=2), encoding="utf-8")
            self._index_artifact(task_id, "strategies", strategies_path)
            self.store.update_progress(task_id, "strategy_generation", JobStatus.SUCCEEDED,
                                       message=f"已生成 {len(strategies)} 套候选策略")
        if self._is_paused(task_id): self.store.update_task(task_id, JobStatus.PAUSED); raise _TaskPaused()

        if not strategies:
            raise RuntimeError("无可用策略：请检查策略生成日志")

        # ── 子阶段3: 全排列投票 ──
        reviewer, aggregator = TournamentReviewer(), VoteAggregator()
        if _sub_phase_done("strategy_voting"):
            evaluation_path = task_dir / "evaluation.json"
            if evaluation_path.is_file():
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                for result in evaluation.get("results", []):
                    aggregator.add_result(result)
            else:
                evaluation = {"ranking": [{"strategy_name": s.get("strategy_name", "")} for s in strategies],
                              "diagnostics": {}}
        else:
            match_count = len(list(itertools.combinations(strategies, 2)))
            self.store.update_progress(task_id, "strategy_voting", JobStatus.RUNNING,
                                       message=f"正在执行 {match_count} 组全排列对比")
            for pair_index, (a, b) in enumerate(itertools.combinations(strategies, 2), 1):
                match_id = f"match-{pair_index:03d}"
                with ThreadPoolExecutor(max_workers=len(REVIEWER_CONFIGS)) as pool:
                    futures = [pool.submit(reviewer.compare_strategies, a, b, research, request.query, binding_rules,
                                           reviewer_id=cfg["id"], match_id=match_id) for cfg in REVIEWER_CONFIGS]
                    for future in as_completed(futures):
                        aggregator.add_result(future.result())
            ranking = aggregator.rank(strategies)
            diagnostics = aggregator.generate_diagnostic(top_n=4)
            evaluation = {"results": aggregator.results, "ranking": ranking, "diagnostics": diagnostics}
            evaluation_path = task_dir / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
            self._index_artifact(task_id, "strategy_evaluation", evaluation_path)
            self.store.update_progress(task_id, "strategy_voting", JobStatus.SUCCEEDED,
                                       message=f"投票完成，已产生 {len(ranking)} 项排名")
        ranking = evaluation.get("ranking", [])
        diagnostics = evaluation.get("diagnostics", {})
        if self._is_paused(task_id): self.store.update_task(task_id, JobStatus.PAUSED); raise _TaskPaused()

        # ── 子阶段4: 策略进化 ──
        if _sub_phase_done("strategy_evolution"):
            evolved_path = task_dir / "evolved_strategies.json"
            evolved = json.loads(evolved_path.read_text(encoding="utf-8")) if evolved_path.is_file() else strategies
        else:
            diagnostic_map = {}
            for item in ranking[:4]:
                name = item["strategy_name"]
                strategy = next((s for s in strategies if s["strategy_name"] == name), None)
                if strategy:
                    diagnostic_map[name] = aggregator.prepare_evolution_input(strategy, name)["diagnosis"]
            self.store.update_progress(task_id, "strategy_evolution", JobStatus.RUNNING,
                                       message="正在进化投票排名最高的策略")
            evolved = StrategyEvolver().evolve_top_n(
                strategies, diagnostic_map, [], research, request.query, n=4, prior_knowledge=binding_rules,
            )
            evolved_path = task_dir / "evolved_strategies.json"
            evolved_path.write_text(json.dumps(evolved, ensure_ascii=False, indent=2), encoding="utf-8")
            self._index_artifact(task_id, "evolved_strategies", evolved_path)
            self.store.update_progress(task_id, "strategy_evolution", JobStatus.SUCCEEDED,
                                       message=f"已进化 {len(evolved)} 套策略")
        return {"ranked_names": [item["strategy_name"] for item in ranking], "evolved_strategies": evolved,
                "execution_context": execution_context, "research": research}

    def _read_input_manifest(self, request: TaskRequest) -> InputManifest:
        if not request.input_manifest_path:
            raise ValueError("task is missing InputManifest v1")
        return InputManifest.model_validate_json(Path(request.input_manifest_path).read_text(encoding="utf-8"))

    def _migrate_legacy_request(self, task_id: str, request: TaskRequest, task_dir: Path) -> TaskRequest:
        if not request.library_path or not request.protein_path:
            raise ValueError("legacy task cannot be resumed because its persisted protein or library input is missing")
        library = migrate_legacy_library(Path(request.library_path), task_dir / "inputs" / "legacy_migrated.smi")
        protein = Path(request.protein_path)
        manifest_path = task_dir / "input_manifest.json"
        manifest = InputManifest(
            query=request.query,
            library_asset=LibraryAsset(source="user", path=str(library), sha256=sha256_file(library),
                                       original_filename=Path(request.library_path).name),
            target_asset=TargetAsset(source="user", locked=True, path=str(protein), sha256=sha256_file(protein),
                                     original_filename=protein.name),
            expert_pocket=request.pocket,
            warnings=["该任务由旧输入格式迁移；新任务仅接受 molecule_id<TAB>SMILES。"],
            constraint_summary=["legacy assets are locked", "pocket coordinates remain deterministic tool outputs"],
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        migrated = request.model_copy(update={"library_path": str(library), "input_manifest_path": str(manifest_path)})
        (task_dir / "request.json").write_text(migrated.model_dump_json(indent=2), encoding="utf-8")
        self.store.update_task_request(task_id, migrated.model_dump(mode="json"))
        return migrated

    def _write_input_manifest(self, request: TaskRequest, manifest: InputManifest) -> None:
        if not request.input_manifest_path:
            raise ValueError("task is missing InputManifest v1")
        Path(request.input_manifest_path).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def _lock_research_identity(self, task_id: str, request: TaskRequest,
                                manifest: InputManifest, research: dict[str, Any],
                                task_dir: Path) -> None:
        """Persist the verified identity so resumes cannot silently change targets."""
        identity = TargetIdentity.model_validate(research["identity"])
        intent = TargetIntent.model_validate(research["intent"])
        request.target_identity = identity
        request.screening_intent = intent
        updated_manifest = manifest.model_copy(update={
            "target_identity": identity,
            "screening_intent": intent,
        })
        self._write_input_manifest(request, updated_manifest)
        (task_dir / "request.json").write_text(request.model_dump_json(indent=2), encoding="utf-8")
        self.store.update_task_request(task_id, request.model_dump(mode="json"))
        manifest.target_identity = identity
        manifest.screening_intent = intent

    def _update_library_manifest(self, request: TaskRequest, validation: dict) -> InputManifest:
        manifest = self._read_input_manifest(request)
        asset = manifest.library_asset.model_copy(update={
            "normalized_path": str(validation["normalized_library"]),
            "total_records": int(validation["total_records"]),
            "accepted_records": int(validation["accepted_records"]),
            "quarantined_records": int(validation["quarantined_records"]),
        })
        updated = manifest.model_copy(update={"library_asset": asset})
        self._write_input_manifest(request, updated)
        return updated

    def _execution_context(self, manifest: InputManifest) -> dict[str, Any]:
        capabilities = [{"action_type": item.action_type.value, "availability": item.availability,
                         "executor": item.executor} for item in list_capabilities(self.settings)]
        return {
            "context_version": "1.0",
            "library": {"binding": "screening_library", "source": manifest.library_asset.source,
                        "locked": True, "format": manifest.library_asset.format,
                        "molecule_count": manifest.library_asset.accepted_records},
            "target": {"binding": "target_structure", "source": manifest.target_asset.source,
                       "locked": manifest.target_asset.locked,
                       "acquisition": "forbidden" if manifest.target_asset.locked else "verified_rcsb_holo_only"},
            "pocket": {"coordinates_owned_by": "deterministic_tools",
                       "user_center_supplied": manifest.expert_pocket.center is not None},
            "invariants": manifest.constraint_summary,
            "capabilities": capabilities,
        }

    def _index_artifact(self, task_id: str, name: str, path: Path) -> None:
        checksum = sha256_file(path)
        if any(item["name"] == name and item["sha256"] == checksum
               for item in self.store.list_artifacts(task_id)):
            return
        self.store.add_artifact(task_id, None, name, path, path.suffix.lstrip(".").upper(), checksum)

    def _run_step(self, task_id: str, step: WorkflowStep, inputs: dict) -> dict:
        phase_id = ACTION_PHASE.get(step.action_type)
        if phase_id:
            self.store.update_progress(
                task_id, phase_id, JobStatus.RUNNING,
                message=f"正在执行 {step.step_id}",
                metadata={"step_id": step.step_id, "action_type": step.action_type.value},
            )
        job = self.tools.submit(task_id, step, inputs, background=False)
        completed = self.store.get_job(job.job_id)
        if not completed or completed.status != JobStatus.SUCCEEDED:
            if phase_id:
                self.store.update_progress(
                    task_id, phase_id, JobStatus.FAILED,
                    message=f"工具步骤 {step.step_id} 失败",
                    error=completed.message if completed else f"step {step.step_id} disappeared",
                    metadata={"step_id": step.step_id, "job_id": job.job_id,
                              "action_type": step.action_type.value},
                )
            raise RuntimeError(completed.message if completed else f"step {step.step_id} disappeared")
        if phase_id:
            self.store.update_progress(
                task_id, phase_id, JobStatus.SUCCEEDED,
                message=f"已完成 {step.step_id}",
                metadata={"step_id": step.step_id, "job_id": job.job_id,
                          "action_type": step.action_type.value},
            )
        try:
            return json.loads(completed.message)
        except json.JSONDecodeError:
            return {}

def build_cpu_baseline_plan() -> WorkflowPlan:
    actions = [
        ("input-validation", ActionType.INPUT_VALIDATION),
        ("pocket-definition", ActionType.POCKET_DEFINITION),
        ("molecule-preparation", ActionType.MOLECULE_STANDARDIZATION),
        ("protein-preparation", ActionType.PROTEIN_PREPARATION),
        ("smina-docking", ActionType.MOLECULAR_DOCKING),
        ("pose-extraction", ActionType.POSE_EXTRACTION),
        ("plip-analysis", ActionType.INTERACTION_ANALYSIS),
        ("final-ranking", ActionType.FINAL_RANKING),
        ("report-generation", ActionType.REPORT_GENERATION),
    ]
    steps = []
    for index, (step_id, action) in enumerate(actions):
        kwargs: dict[str, Any] = {}
        if action == ActionType.MOLECULAR_DOCKING:
            kwargs["parameters"] = {"exhaustiveness": 4, "num_modes": 3}
            from autovs.schemas import ResourceProfile
            kwargs["resource_profile"] = ResourceProfile(executor="conda", environment="smina_stage2", cpus=10, timeout_seconds=259200)
        steps.append(WorkflowStep(step_id=step_id, action_type=action,
                                  requires=[steps[-1].step_id] if steps else [], **kwargs))
    return WorkflowPlan(strategy_id="cpu-noncovalent-sbdd-baseline", steps=steps)
