from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from autovs.capabilities import health_report
from autovs.compiler import choose_executable_strategy, compile_strategy
from autovs.config import Settings, load_settings
from autovs.db import StateStore
from autovs.reporting import generate_failure_report, generate_report
from autovs.schemas import ActionType, JobStatus, PocketResolution, TaskRequest, WorkflowPlan, WorkflowStep
from autovs.security import sha256_file
from autovs.tool_manager import ToolManager


PIPELINE_PHASES = [
    ("input_validation", "输入校验"),
    ("target_research", "靶点调研"),
    ("pocket_definition", "口袋确定"),
    ("strategy_generation", "策略生成"),
    ("strategy_voting", "全排列投票"),
    ("strategy_evolution", "策略进化"),
    ("strategy_selection", "可执行策略选择"),
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
    ActionType.POCKET_DEFINITION: "pocket_definition",
    ActionType.MOLECULE_STANDARDIZATION: "molecule_standardization",
    ActionType.CONFORMER_GENERATION: "molecule_standardization",
    ActionType.PHYSICOCHEMICAL_FILTERING: "molecule_standardization",
    ActionType.PROTEIN_PREPARATION: "protein_preparation",
    ActionType.MOLECULAR_DOCKING: "molecular_docking",
    ActionType.POSE_EXTRACTION: "pose_extraction",
    ActionType.INTERACTION_ANALYSIS: "interaction_analysis",
    ActionType.FINAL_RANKING: "final_ranking",
}

LLM_ONLY_PHASES = ("target_research", "strategy_generation", "strategy_voting", "strategy_evolution")


class PipelineService:
    """The single application service used by CLI, Web, and tests."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.store = StateStore(self.settings.database_path)
        self.tools = ToolManager(self.settings, self.store)

    def submit(self, request: TaskRequest, *, use_llm_planning: bool = True) -> str:
        staged, task_dir = self._stage_request(request)
        task_id = self.store.create_task(staged.model_dump(mode="json"), task_dir)
        self._initialize_progress(task_id, use_llm_planning)
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
        if task["status"] == JobStatus.RUNNING.value:
            raise ValueError("task is already running")
        threading.Thread(target=self._run_existing, args=(task_id, True), daemon=True).start()

    def get_task(self, task_id: str) -> dict | None:
        task = self.store.get_task(task_id)
        if task:
            task["jobs"] = self.store.list_jobs(task_id)
            task["artifacts"] = self.store.list_artifacts(task_id)
            task["progress"] = self.store.list_progress(task_id)
        return task

    def _initialize_progress(self, task_id: str, use_llm_planning: bool) -> None:
        self.store.initialize_progress(task_id, PIPELINE_PHASES)
        if not use_llm_planning:
            for phase_id in LLM_ONLY_PHASES:
                self.store.update_progress(
                    task_id, phase_id, JobStatus.SKIPPED,
                    message="基础链路诊断模式跳过 LLM 规划",
                )

    def _stage_request(self, request: TaskRequest) -> tuple[TaskRequest, Path]:
        source_protein = Path(request.protein_path).expanduser().resolve()
        source_library = Path(request.library_path).expanduser().resolve()
        if not source_protein.is_file() or not source_library.is_file():
            raise ValueError("protein_path and library_path must be existing files")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fingerprint = hashlib.sha256(f"{request.query}|{source_protein}|{source_library}".encode()).hexdigest()[:8]
        task_dir = self.settings.task_root / f"task_{stamp}_{fingerprint}"
        inputs = task_dir / "inputs"; inputs.mkdir(parents=True, exist_ok=False)
        protein = inputs / f"protein{source_protein.suffix.lower()}"
        library = inputs / f"library{source_library.suffix.lower()}"
        shutil.copy2(source_protein, protein); shutil.copy2(source_library, library)
        staged = request.model_copy(update={"protein_path": str(protein), "library_path": str(library)})
        (task_dir / "request.json").write_text(staged.model_dump_json(indent=2), encoding="utf-8")
        return staged, task_dir

    def _run_existing(self, task_id: str, use_llm_planning: bool) -> None:
        task = self.store.get_task(task_id)
        if not task:
            return
        self.store.update_task(task_id, JobStatus.RUNNING)
        task_dir = Path(task["task_dir"])
        request = TaskRequest.model_validate(task["request"])
        rejected: list[dict] = []
        try:
            self._run_step(
                task_id, WorkflowStep(step_id="input-validation", action_type=ActionType.INPUT_VALIDATION),
                {"protein_path": request.protein_path, "library_path": request.library_path},
            )
            if use_llm_planning:
                planning = self._run_planning(task_id, request, task_dir)
                self.store.update_progress(task_id, "strategy_selection", JobStatus.RUNNING,
                                           message="按投票排名校验候选策略")
                _, plan, rejected = choose_executable_strategy(planning["ranked_names"], planning["evolved_strategies"])
                self.store.update_progress(
                    task_id, "strategy_selection", JobStatus.SUCCEEDED,
                    message=f"已选择可执行策略：{plan.strategy_id}",
                    metadata={"strategy_id": plan.strategy_id, "rejected_count": len(rejected)},
                )
            else:
                plan = build_cpu_baseline_plan()
                pocket_resolution = self._resolve_pocket_preflight(task_id, request, task_dir, {})
                planning = {"mode": "deterministic_cpu_baseline", "ranked_names": [plan.strategy_id],
                            "pocket_resolution": pocket_resolution.model_dump(mode="json")}
                self.store.update_progress(task_id, "strategy_selection", JobStatus.SUCCEEDED,
                                           message="已选择确定性 CPU 基线策略",
                                           metadata={"strategy_id": plan.strategy_id})
            (task_dir / "workflow_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            result = self._execute_core(task_id, request, plan, rejected, planning)
            self.store.finish_pending_progress(task_id, message="未包含在本次可执行策略中")
            self.store.update_task(task_id, JobStatus.SUCCEEDED, result=result)
        except Exception as exc:
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

    def _run_planning(self, task_id: str, request: TaskRequest, task_dir: Path) -> dict:
        from src.agents.expert_committee import REVIEWER_CONFIGS, TournamentReviewer
        from src.agents.judge_agent import VoteAggregator
        from src.agents.strategy_evolver import StrategyEvolver
        from src.agents.strategy_generator import StrategyGeneratorAgent
        from src.agents.target_scout import TargetScoutAgent

        self.store.update_progress(task_id, "target_research", JobStatus.RUNNING,
                                   message="正在检索并验证靶点结构证据")
        research = TargetScoutAgent().deep_research(request.query)
        research["_user_query"] = request.query
        (task_dir / "research.json").write_text(json.dumps(research, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.store.update_progress(task_id, "target_research", JobStatus.SUCCEEDED,
                                   message="靶点调研已完成，证据已固化为 research.json")
        pocket_resolution = self._resolve_pocket_preflight(task_id, request, task_dir, research)
        # Strategy agents may interpret this verified result, but the executor never accepts
        # coordinates copied back from an LLM-authored strategy.
        research["resolved_pocket"] = pocket_resolution.model_dump(mode="json")
        self.store.update_progress(task_id, "strategy_generation", JobStatus.RUNNING,
                                   message="正在生成结构化虚拟筛选策略")
        strategies = StrategyGeneratorAgent().generate_strategies(research)["strategies"]
        (task_dir / "strategies.json").write_text(json.dumps(strategies, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.update_progress(task_id, "strategy_generation", JobStatus.SUCCEEDED,
                                   message=f"已生成 {len(strategies)} 套候选策略")

        reviewer, aggregator = TournamentReviewer(), VoteAggregator()
        match_count = len(list(itertools.combinations(strategies, 2)))
        self.store.update_progress(task_id, "strategy_voting", JobStatus.RUNNING,
                                   message=f"正在执行 {match_count} 组全排列对比")
        for pair_index, (a, b) in enumerate(itertools.combinations(strategies, 2), 1):
            match_id = f"match-{pair_index:03d}"
            with ThreadPoolExecutor(max_workers=len(REVIEWER_CONFIGS)) as pool:
                futures = [pool.submit(reviewer.compare_strategies, a, b, research, request.query, "",
                                       reviewer_id=cfg["id"], match_id=match_id) for cfg in REVIEWER_CONFIGS]
                for future in as_completed(futures):
                    aggregator.add_result(future.result())
        ranking = aggregator.rank(strategies)
        diagnostics = aggregator.generate_diagnostic(top_n=4)
        evaluation = {"results": aggregator.results, "ranking": ranking, "diagnostics": diagnostics}
        (task_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.update_progress(task_id, "strategy_voting", JobStatus.SUCCEEDED,
                                   message=f"投票完成，已产生 {len(ranking)} 项排名")
        diagnostic_map = {}
        for item in ranking[:4]:
            name = item["strategy_name"]
            strategy = next(s for s in strategies if s["strategy_name"] == name)
            diagnostic_map[name] = aggregator.prepare_evolution_input(strategy, name)["diagnosis"]
        self.store.update_progress(task_id, "strategy_evolution", JobStatus.RUNNING,
                                   message="正在进化投票排名最高的策略")
        evolved = StrategyEvolver().evolve_top_n(strategies, diagnostic_map, [], research, request.query, n=4)
        (task_dir / "evolved_strategies.json").write_text(json.dumps(evolved, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.update_progress(task_id, "strategy_evolution", JobStatus.SUCCEEDED,
                                   message=f"已进化 {len(evolved)} 套策略")
        return {"ranked_names": [item["strategy_name"] for item in ranking], "evolved_strategies": evolved,
                "pocket_resolution": pocket_resolution.model_dump(mode="json")}

    def _resolve_pocket_preflight(self, task_id: str, request: TaskRequest, task_dir: Path,
                                  research: dict) -> PocketResolution:
        step = WorkflowStep(step_id="pocket-definition", action_type=ActionType.POCKET_DEFINITION,
                            requires=["input-validation"])
        inputs = {
            "protein_path": request.protein_path,
            "center": request.pocket.center,
            "size": request.pocket.size,
            "key_residues": request.pocket.key_residues,
            "cocrystal_ligand": request.pocket.cocrystal_ligand,
        }
        research_path = task_dir / "research.json"
        if research:
            if not research_path.exists():
                research_path.write_text(json.dumps(research, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            inputs["research_path"] = str(research_path)
        output = self._run_step(task_id, step, inputs)
        resolution = PocketResolution.model_validate_json(Path(output["pocket"]).read_text(encoding="utf-8"))
        (task_dir / "pocket_resolution.json").write_text(resolution.model_dump_json(indent=2), encoding="utf-8")
        return resolution

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

    def _execute_core(self, task_id: str, request: TaskRequest, plan: WorkflowPlan,
                      rejected: list[dict], planning: dict) -> dict:
        task = self.store.get_task(task_id); assert task
        task_dir = Path(task["task_dir"])
        by_action: dict[ActionType, WorkflowStep] = {}
        for step in plan.steps:
            by_action.setdefault(step.action_type, step)

        validation = self._run_step(task_id, by_action.get(ActionType.INPUT_VALIDATION, WorkflowStep(step_id="input-validation", action_type=ActionType.INPUT_VALIDATION)),
                                    {"protein_path": request.protein_path, "library_path": request.library_path})
        pocket_resolution = PocketResolution.model_validate(planning["pocket_resolution"])
        pocket_data = pocket_resolution.selected_pocket
        prep_step = by_action.get(ActionType.MOLECULE_STANDARDIZATION) or WorkflowStep(step_id="molecule-preparation", action_type=ActionType.MOLECULE_STANDARDIZATION)
        prepared = self._run_step(task_id, prep_step, {"library_path": request.library_path})
        protein = self._run_step(task_id, by_action.get(ActionType.PROTEIN_PREPARATION, WorkflowStep(step_id="protein-preparation", action_type=ActionType.PROTEIN_PREPARATION)),
                                 {"protein_path": request.protein_path})

        docking_step = by_action.get(ActionType.MOLECULAR_DOCKING)
        if not docking_step:
            raise RuntimeError("selected strategy does not contain molecular_docking")
        docking = self._run_step(task_id, docking_step, {
            "receptor_pdbqt": protein["receptor_pdbqt"], "ligands_sdf": prepared["prepared_library"],
            "manifest_csv": prepared["manifest"], "center": pocket_data.center, "size": pocket_data.size,
        })
        score_csv = Path(docking["scores_csv"])
        pose_step = by_action.get(ActionType.POSE_EXTRACTION)
        plip_step = by_action.get(ActionType.INTERACTION_ANALYSIS)
        if pose_step and plip_step:
            poses = self._run_step(task_id, pose_step, {"receptor_pdb": protein["receptor_pdb"],
                                   "docked_poses": docking["docked_poses"], "engine": "smina", "pose_metric": "best_affinity"})
            interactions = self._run_step(task_id, plip_step, {"complex_index": poses["complex_index"],
                                          "key_residues": request.pocket.key_residues})
            score_csv = _merge_score_csvs(score_csv, Path(interactions["plip_scores"]), task_dir / "combined_scores.csv")
        ranking_step = by_action.get(ActionType.FINAL_RANKING) or WorkflowStep(step_id="final-ranking", action_type=ActionType.FINAL_RANKING)
        ranked_output = self._run_step(task_id, ranking_step, {"scores_csv": str(score_csv)})
        import csv
        with Path(ranked_output["top_hits"]).open(encoding="utf-8-sig", newline="") as handle:
            top_hits = list(csv.DictReader(handle))

        # ADMET/PLIP/MD are explicit evidence gaps until their production adapters finish successfully.
        evidence_gaps = []
        for action in (ActionType.INTERACTION_ANALYSIS, ActionType.ADMET_FILTERING, ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS):
            if action not in by_action or (request.cpu_only and action in {ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS}):
                evidence_gaps.append(action.value)
        artifacts = self.store.list_artifacts(task_id)
        self.store.update_progress(task_id, "report_generation", JobStatus.RUNNING,
                                   message="正在汇总结果、版本与可追溯证据")
        reports = generate_report(task_id, task_dir, request=request.model_dump(mode="json"), plan=plan.model_dump(mode="json"),
                                  results=top_hits, rejected_strategies=rejected, health=health_report(self.settings),
                                  jobs=self.store.list_jobs(task_id), artifacts=artifacts,
                                  pocket_resolution=pocket_resolution.model_dump(mode="json"))
        for name, raw_path in reports.items():
            path = Path(raw_path); self.store.add_artifact(task_id, None, name, path, path.suffix.lstrip(".").upper(), sha256_file(path))
        self.store.update_progress(task_id, "report_generation", JobStatus.SUCCEEDED,
                                   message="可复现报告已生成")
        return {"task_id": task_id, "status": "succeeded", "top_hits": top_hits, "reports": reports,
                "workflow_plan": str(task_dir / "workflow_plan.json"), "evidence_gaps": evidence_gaps,
                "rejected_strategies": rejected, "planning": planning,
                "pocket_resolution": pocket_resolution.model_dump(mode="json")}


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


def _merge_score_csvs(primary: Path, additional: Path, output: Path) -> Path:
    import csv
    with primary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with additional.open(encoding="utf-8-sig", newline="") as handle:
        extra = {row["source_id"]: row for row in csv.DictReader(handle)}
    for row in rows:
        row.update(extra.get(row.get("source_id", ""), {}))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return output
