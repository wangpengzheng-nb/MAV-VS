from __future__ import annotations

import csv
import hashlib
import json
import threading
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from autovs.capabilities import list_capabilities
from autovs.af3 import ToolPending
from autovs.config import Settings
from autovs.db import StateStore
from autovs.library import normalize_smi_library, verify_default_library
from autovs.pocket import resolve_pocket
from autovs.preparation import prepare_library
from autovs.ranking import rank_csv
from autovs.schemas import (
    ActionType, ExecutorConfig, ExecutorType, InputManifest, JobRecord, JobStatus,
    WorkflowStep,
)
from autovs.security import ensure_within, run_argv, sha256_file
from autovs.structure_acquisition import acquire_rcsb_structures


class ToolManager:
    def __init__(self, settings: Settings, store: StateStore):
        self.settings, self.store = settings, store

    @property
    def allowed_roots(self) -> list[Path]:
        return [self.settings.task_root, self.settings.config_path.parent.parent]

    # ── 统一工具执行器 API ──────────────────────────────────────────

    def _executor(self, name: str) -> ExecutorConfig | None:
        """查找工具执行器配置。"""
        return self.settings.executor_config(name)

    def _require_executor(self, name: str) -> ExecutorConfig:
        """查找工具执行器配置，找不到或路径无效时抛出明确错误。"""
        cfg = self._executor(name)
        if cfg is None:
            raise RuntimeError(f"工具 '{name}' 未在 config/tools.toml [executors] 中注册")
        if cfg.executor == ExecutorType.SUBPROCESS:
            if not cfg.path:
                raise RuntimeError(f"工具 '{name}' 的 path 为空")
            if not Path(cfg.path).exists():
                raise RuntimeError(
                    f"工具 '{name}' 的二进制不存在: {cfg.path}"
                    + (f"（来自 conda 环境 {cfg.env_hint}）" if cfg.env_hint else "")
                )
        elif cfg.executor == ExecutorType.APPTAINER:
            resolved = cfg.resolved_path(str(self.settings.config_path.parent))
            if not resolved or not resolved.exists():
                raise RuntimeError(f"工具 '{name}' 的容器镜像不存在: {cfg.path}")
        return cfg

    def health_check(self, name: str) -> tuple[bool, str]:
        """快速健康检查：工具是否可用。

        Returns:
            (healthy, message)
        """
        cfg = self._executor(name)
        if cfg is None:
            return False, f"工具 '{name}' 未注册"
        try:
            if cfg.executor == ExecutorType.SUBPROCESS:
                if not cfg.path or not Path(cfg.path).exists():
                    return False, f"二进制不存在: {cfg.path}"
                if cfg.health_check and cfg.health_check != "--version":
                    result = run_argv(
                        [cfg.path, cfg.health_check],
                        cwd=Path("/tmp"), timeout=10,
                    )
                    if result.returncode != 0:
                        return False, f"健康检查失败 (exit={result.returncode}): {result.stderr[:200]}"
                return True, f"二进制可用: {cfg.path}"
            elif cfg.executor == ExecutorType.PYTHON_MODULE:
                if not cfg.module:
                    return False, "未配置 Python 模块名"
                import importlib.util
                if importlib.util.find_spec(cfg.module) is not None:
                    return True, f"Python 模块可导入: {cfg.module}"
                return False, f"Python 模块不可导入: {cfg.module}（可能需要 conda activate {cfg.env}）"
            elif cfg.executor == ExecutorType.APPTAINER:
                resolved = cfg.resolved_path(str(self.settings.config_path.parent))
                if not resolved or not resolved.exists():
                    return False, f"容器镜像不存在: {cfg.path}"
                return True, f"容器镜像可用: {resolved}"
            return False, f"未知执行器类型: {cfg.executor.value}"
        except Exception as exc:
            return False, f"健康检查异常: {exc}"

    def _required_binary(self, name: str) -> Path:
        """【已废弃】获取工具二进制路径；新代码应使用 _require_executor()。

        保留此方法以确保现有 _dispatch 分支无回归。
        """
        cfg = self._require_executor(name)
        return Path(cfg.path) if cfg.path else Path()

    def submit(self, task_id: str, step: WorkflowStep, inputs: dict[str, Any], *, background: bool = True) -> JobRecord:
        task = self.store.get_task(task_id)
        if not task:
            raise ValueError(f"unknown task_id: {task_id}")
        checkpoint_key = self._checkpoint_key(step, inputs)
        existing = self.store.find_checkpoint_job(task_id, step.step_id, checkpoint_key)
        if existing and self._checkpoint_outputs_exist(existing.message):
            return existing
        job = self.store.create_job(task_id, step.step_id, step.action_type, command={"checkpoint_key": checkpoint_key})
        if background:
            threading.Thread(target=self._execute, args=(job.job_id, task, step, inputs), daemon=True).start()
        else:
            self._execute(job.job_id, task, step, inputs)
        return self.store.get_job(job.job_id)  # type: ignore[return-value]

    def _checkpoint_key(self, step: WorkflowStep, inputs: dict[str, Any]) -> str:
        normalized = {"step": step.model_dump(mode="json"), "inputs": {}}
        for key, value in sorted(inputs.items()):
            if isinstance(value, str):
                candidate = Path(value)
                normalized["inputs"][key] = {"path": value, "sha256": sha256_file(candidate) if candidate.is_file() else None}
            else:
                normalized["inputs"][key] = value
        return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _checkpoint_outputs_exist(message: str) -> bool:
        try:
            outputs = json.loads(message)
        except json.JSONDecodeError:
            return False
        paths = [Path(value) for value in _walk_output_strings(outputs) if "/" in value or "\\" in value]
        return bool(paths) and all(path.exists() for path in paths)

    def _execute(self, job_id: str, task: dict, step: WorkflowStep, inputs: dict[str, Any]) -> None:
        self.store.update_job(job_id, JobStatus.RUNNING)
        task_dir = ensure_within(task["task_dir"], [self.settings.task_root], must_exist=True)
        work_dir = task_dir / "steps" / step.step_id
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            outputs = self._dispatch(step, inputs, work_dir)
            for name, value in outputs.items():
                if isinstance(value, Path) and value.is_file():
                    try:
                        value.resolve().relative_to(task_dir.resolve())
                    except ValueError:
                        continue  # shared read-only assets are referenced by checksum, not exposed as task artifacts
                    self.store.add_artifact(task["task_id"], job_id, name, value, value.suffix.lstrip(".").upper(), sha256_file(value))
            self.store.update_job(job_id, JobStatus.SUCCEEDED, message=json.dumps(_jsonable(outputs), ensure_ascii=False))
        except ToolPending as exc:
            payload = {
                "job_id": job_id,
                "task_id": task["task_id"],
                "step_id": step.step_id,
                "action_type": step.action_type.value,
                "message": str(exc),
                **exc.payload,
            }
            pending_path = exc.state_path or (work_dir / "pending.json")
            if not pending_path.is_file():
                pending_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.add_artifact(task["task_id"], job_id, "pending_state", pending_path,
                                    pending_path.suffix.lstrip(".").upper(), sha256_file(pending_path))
            self.store.update_job(job_id, JobStatus.PAUSED, message=json.dumps(_jsonable({
                **payload,
                "pending_state": pending_path,
            }), ensure_ascii=False), slurm_job_id=str(payload.get("slurm_job_id") or "") or None)
        except Exception as exc:
            failure_path = work_dir / "failure.json"
            failure_path.write_text(json.dumps({
                "job_id": job_id,
                "task_id": task["task_id"],
                "step_id": step.step_id,
                "action_type": step.action_type.value,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.add_artifact(task["task_id"], job_id, "failure_diagnostic", failure_path,
                                    "JSON", sha256_file(failure_path))
            for diagnostic in sorted(work_dir.rglob("*")):
                if (not diagnostic.is_file() or diagnostic == failure_path
                        or diagnostic.suffix.lower() not in {".log", ".txt", ".json"}):
                    continue
                relative_name = diagnostic.relative_to(work_dir).as_posix().replace("/", "__")
                self.store.add_artifact(task["task_id"], job_id, f"diagnostic__{relative_name}",
                                        diagnostic, diagnostic.suffix.lstrip(".").upper(), sha256_file(diagnostic))
            self.store.update_job(job_id, JobStatus.FAILED, message=f"{type(exc).__name__}: {exc}")

    def _dispatch(self, step: WorkflowStep, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        action = step.action_type
        if action == ActionType.INPUT_VALIDATION:
            unexpected = set(inputs) - {"protein_path", "library_path", "input_manifest_path"}
            if unexpected:
                raise ValueError(f"input_validation received unsupported inputs: {sorted(unexpected)}")
            library = ensure_within(inputs["library_path"], self.allowed_roots, must_exist=True)
            manifest_path = ensure_within(inputs["input_manifest_path"], self.allowed_roots, must_exist=True)
            manifest = InputManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            protein = None
            if inputs.get("protein_path"):
                protein = ensure_within(inputs["protein_path"], self.allowed_roots, must_exist=True)
                if protein.suffix.lower() != ".pdb":
                    raise ValueError("protein must be a preprocessed PDB file")
                if not any(line.startswith("ATOM  ") for line in protein.read_text(errors="ignore").splitlines()):
                    raise ValueError("protein PDB contains no ATOM records")
            max_molecules = int(self.settings.limit("max_library_molecules", 1_000_000))
            if manifest.library_asset.source == "builtin":
                cfg = self.settings.library()
                check = verify_default_library(
                    library, str(cfg.get("sha256", "")), int(cfg.get("molecule_count", 0)),
                )
                if check["status"] != "available":
                    raise ValueError(str(check.get("reason", "default library is unavailable")))
                validation = work_dir / "library_validation.json"
                rejected = work_dir / "library_rejected.tsv"
                payload = {
                    "format": "strict_smi_v1", "source": "builtin", "version": cfg.get("version"),
                    "input_path": str(library), "input_sha256": sha256_file(library),
                    "normalized_path": str(library), "normalized_sha256": sha256_file(library),
                    "total_records": int(cfg["molecule_count"]), "accepted_records": int(cfg["molecule_count"]),
                    "quarantined_records": 0, "rejection_reasons": {},
                }
                validation.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                rejected.write_text("line_number\tmolecule_id\tsmiles\treason\n", encoding="utf-8")
                normalized: dict[str, Any] = {**payload, "normalized_library": library,
                                              "validation": validation, "rejected": rejected}
            else:
                normalized = normalize_smi_library(library, work_dir, max_molecules=max_molecules, source="user")
            output = work_dir / "input_validation.json"
            output.write_text(json.dumps({
                "protein": str(protein) if protein else None,
                "protein_sha256": sha256_file(protein) if protein else None,
                "library": str(library), "library_sha256": sha256_file(library),
                "normalized_library": str(normalized["normalized_library"]),
                "normalized_library_sha256": normalized["normalized_sha256"],
                "total_records": normalized["total_records"],
                "accepted_records": normalized["accepted_records"],
                "quarantined_records": normalized["quarantined_records"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"input_validation": output, "normalized_library": normalized["normalized_library"],
                    "library_validation": normalized["validation"], "library_rejected": normalized["rejected"],
                    "total_records": normalized["total_records"], "accepted_records": normalized["accepted_records"],
                    "quarantined_records": normalized["quarantined_records"]}
        if action == ActionType.TARGET_STRUCTURE_ACQUISITION:
            unexpected = set(inputs) - {"research_path", "limit", "selected_strategy_id"}
            if unexpected:
                raise ValueError(f"target_structure_acquisition received unsupported inputs: {sorted(unexpected)}")
            research_path = ensure_within(inputs["research_path"], [work_dir.parent.parent], must_exist=True)
            return acquire_rcsb_structures(
                research_path, work_dir, limit=min(int(inputs.get("limit", 5)), 5),
                selected_strategy_id=str(inputs.get("selected_strategy_id", "")),
            )
        if action == ActionType.TARGET_STRUCTURE_PREDICTION:
            from autovs.af3 import predict_structure
            research_path = ensure_within(inputs["research_path"], [work_dir.parent.parent], must_exist=True)
            return predict_structure(
                research_path=research_path,
                work_dir=work_dir,
                parameters=step.parameters | {k: v for k, v in inputs.items() if k != "research_path"},
            )
        if action == ActionType.POCKET_DEFINITION:
            protein = ensure_within(inputs["protein_path"], self.allowed_roots, must_exist=True)
            if "research" in inputs:
                raise ValueError("inline research is forbidden; use a checksummed research_path from the task directory")
            research: dict[str, Any] = {}
            research_path = inputs.get("research_path")
            if research_path:
                task_dir = work_dir.parent.parent
                verified_research = ensure_within(research_path, [task_dir], must_exist=True)
                research = json.loads(verified_research.read_text(encoding="utf-8"))
                if not isinstance(research, dict):
                    raise ValueError("research artifact must contain a JSON object")
            plip_cfg = self._executor("plip")
            plip_path = plip_cfg.resolved_path() if plip_cfg and plip_cfg.path and Path(plip_cfg.path).exists() else None
            pocket = resolve_pocket(protein, center=inputs.get("center"), size=tuple(inputs.get("size", (24, 24, 24))),
                                    key_residues=list(inputs.get("key_residues", [])),
                                    cocrystal_ligand=inputs.get("cocrystal_ligand"), research=research,
                                    work_dir=work_dir, plip_path=plip_path)
            output = work_dir / "pocket.json"
            output.write_text(pocket.model_dump_json(indent=2), encoding="utf-8")
            return {"pocket": output}
        if action in {ActionType.MOLECULE_STANDARDIZATION, ActionType.CONFORMER_GENERATION, ActionType.PHYSICOCHEMICAL_FILTERING}:
            library = ensure_within(inputs["library_path"], self.allowed_roots, must_exist=True)
            return prepare_library(library, work_dir, max_molecules=int(self.settings.limit("max_library_molecules", 1_000_000)),
                                   mw_range=tuple(step.parameters.get("mw_range", (150, 800))),
                                   logp_range=tuple(step.parameters.get("logp_range", (-2, 8))))
        if action == ActionType.PROTEIN_PREPARATION:
            protein = ensure_within(inputs["protein_path"], self.allowed_roots, must_exist=True)
            obabel_cfg = self._require_executor("obabel")
            obabel = Path(obabel_cfg.path) if obabel_cfg.path else None
            if not obabel or not obabel.exists():
                raise RuntimeError("OpenBabel 不可用：请确认 config/tools.toml [executors.obabel]")
            clean, pdbqt = work_dir / "receptor_clean.pdb", work_dir / "receptor.pdbqt"
            protein_only = work_dir / "protein_only_input.pdb"
            protein_lines = [line for line in protein.read_text(errors="ignore").splitlines() if line.startswith("ATOM  ")]
            if not protein_lines:
                raise ValueError("protein preparation found no ATOM records")
            protein_only.write_text("\n".join(protein_lines) + "\nEND\n", encoding="utf-8")
            result = run_argv([str(obabel), "-ipdb", str(protein_only), "-opdb", "-O", str(clean), "-h"], cwd=work_dir, timeout=1800, log_path=work_dir / "protein_prep.log")
            if result.returncode:
                raise RuntimeError(f"OpenBabel protein preparation failed: {result.stderr[-500:]}")
            result = run_argv([str(obabel), "-ipdb", str(clean), "-opdbqt", "-O", str(pdbqt), "-xr"], cwd=work_dir, timeout=1800, log_path=work_dir / "pdbqt.log")
            if result.returncode:
                raise RuntimeError(f"PDBQT conversion failed: {result.stderr[-500:]}")
            return {"receptor_pdb": clean, "receptor_pdbqt": pdbqt}
        if action == ActionType.FINAL_RANKING:
            score_csv = ensure_within(inputs["scores_csv"], [self.settings.task_root], must_exist=True)
            output = work_dir / "top20.csv"
            rows = rank_csv(score_csv, output, top_n=int(self.settings.limit("final_hits", 20)))
            return {"top_hits": output, "hit_count": len(rows)}
        if action == ActionType.MOLECULAR_DOCKING:
            return self._run_docking(step, inputs, work_dir)
        if action == ActionType.DIVERSITY_SELECTION:
            return self._run_diversity(inputs, work_dir)
        if action == ActionType.POSE_EXTRACTION:
            return self._extract_poses(inputs, work_dir)
        if action == ActionType.INTERACTION_ANALYSIS:
            return self._run_plip(inputs, work_dir)
        if action == ActionType.STRUCTURE_ANALYSIS:
            return self._analyze_structure(inputs, work_dir)
        if action == ActionType.PROTEIN_REPAIR:
            return self._repair_protein(inputs, work_dir)
        if action == ActionType.PROTONATION:
            return self._protonate(inputs, work_dir)
        if action == ActionType.MOLECULE_STANDARDIZATION_V2:
            return self._standardize_v2(inputs, work_dir)
        if action == ActionType.LIGAND_3D_ENUMERATION:
            return self._enumerate_3d(inputs, work_dir)
        if action == ActionType.IONIZATION_ENUMERATION:
            return self._enumerate_ionization(inputs, work_dir)
        if action == ActionType.PDBQT_PARAMETERIZATION:
            return self._prepare_pdbqt(inputs, work_dir)
        if action == ActionType.FORMAT_CONVERSION:
            return self._convert_format(inputs, work_dir)
        if action == ActionType.ADMET_FILTERING:
            return self._run_admet(inputs, work_dir)
        if action == ActionType.POSE_VALIDATION:
            return self._run_pose_validation(inputs, work_dir)
        if action == ActionType.POCKET_PREDICTION:
            return self._run_pocket_prediction(inputs, work_dir)
        if action == ActionType.DIFFDOCK_DOCKING:
            return self._run_diffdock(inputs, work_dir)
        if action == ActionType.GEOMETRIC_POCKET_DETECTION:
            return self._run_fpocket(inputs, work_dir)
        if action == ActionType.PHARMACOPHORE_SCREENING:
            return self._run_pharmacophore(inputs, work_dir)
        if action == ActionType.STRUCTURAL_HOMOLOGY_SEARCH:
            return self._run_foldseek(inputs, work_dir)
        if action in {ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS}:
            from autovs.gromacs import submit_gromacs_md
            receptor = ensure_within(inputs["receptor_pdb"], [self.settings.task_root], must_exist=True)
            selected = ensure_within(inputs["selected_poses"], [self.settings.task_root], must_exist=True)
            return submit_gromacs_md(
                receptor_pdb=receptor,
                selected_poses=selected,
                work_dir=work_dir,
                settings=self.settings,
                parameters=step.parameters | inputs,
                short=action == ActionType.SHORT_MD,
            )
        raise RuntimeError(f"{action.value} has no active production adapter; capability is not executable yet")

    # ─── 统一对接引擎路由 ─────────────────────────────────────────────

    def _run_docking(self, step: WorkflowStep, inputs: dict[str, Any],
                     work_dir: Path) -> dict[str, Any]:
        """三引擎统一对接入口.

        根据策略参数或自动选择引擎:
        - smina: CPU快速对接
        - gnina: GPU CNN打分对接
        - diffdock: PPI靶点扩散模型对接
        """
        from autovs.docking import select_docking_engine

        # 读取target_type用于引擎选择
        research: dict[str, Any] = {}
        research_path = self.settings.task_root.parent / "research"
        research_json = work_dir.parent.parent / "research.json"
        if research_json.is_file():
            try:
                research = json.loads(research_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        # GPU可用性检查
        gnina_cfg = self._executor("gnina")
        gpu_available = gnina_cfg is not None and gnina_cfg.gpu_required
        cpu_only = step.parameters.get("cpu_only", False)

        engine = select_docking_engine(
            strategy_params=step.parameters,
            research=research,
            gpu_available=gpu_available,
            cpu_only=cpu_only,
        )

        if engine == "gnina":
            return self._run_gnina(step, inputs, work_dir)
        elif engine == "diffdock":
            return self._run_diffdock(step, inputs, work_dir)
        else:
            return self._run_smina(step, inputs, work_dir)

    # ─── GNINA GPU 对接 ──────────────────────────────────────────────

    def _run_gnina(self, step: WorkflowStep, inputs: dict[str, Any],
                   work_dir: Path) -> dict[str, Any]:
        """GNINA GPU对接 (CNN scoring)."""
        from autovs.docking import submit_gnina_docking, parse_docking_scores

        receptor = ensure_within(
            inputs.get("receptor_pdbqt", inputs.get("receptor_pdb")),
            [self.settings.task_root], must_exist=True,
        )
        ligands = ensure_within(
            inputs.get("ligands_sdf", inputs.get("enumerated_3d_sdf")),
            [self.settings.task_root], must_exist=True,
        )
        center = inputs.get("center")
        size = inputs.get("size", (24, 24, 24))
        if not center or len(center) != 3:
            raise ValueError("docking requires a three-value pocket center")

        gnina_cfg = self._require_executor("gnina")
        gnina_bin = str(gnina_cfg.path) if gnina_cfg.path else "gnina"

        # 从配置读取GPU资源
        gpu_config = dict(self.settings.raw.get("slurm", {}).get("gpu", {}))

        result = submit_gnina_docking(
            receptor_pdbqt=Path(str(receptor)),
            ligands_sdf=Path(str(ligands)),
            center=(float(center[0]), float(center[1]), float(center[2])),
            size=(float(size[0]), float(size[1]), float(size[2])),
            output_dir=work_dir,
            exhaustiveness=step.parameters.get("exhaustiveness", 8),
            num_modes=step.parameters.get("num_modes", 5),
            cnn_scoring=step.parameters.get("cnn_scoring", "rescore"),
            cnn_rotation=step.parameters.get("cnn_rotation", 1),
            seed=step.parameters.get("seed", 61453),
            gnina_bin=gnina_bin,
            gpu_config=gpu_config,
            submit_slurm=step.parameters.get("submit_slurm", True),
        )

        slurm_id = result.get("slurm_job_id", "")
        if slurm_id:
            # Slurm提交: 返回pending状态
            from autovs.af3 import ToolPending
            state_path = work_dir / "gnina_state.json"
            state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            raise ToolPending(
                f"GNINA Slurm job {slurm_id} submitted",
                state_path=state_path,
                slurm_job_id=slurm_id,
            )

        # 本地模式: 直接解析结果
        output_sdf = Path(result["output_sdf"])
        if not output_sdf.is_file():
            raise RuntimeError("GNINA output SDF not found")

        manifest_path = inputs.get("manifest_csv")
        manifest = Path(str(manifest_path)) if manifest_path else None
        scores_csv = parse_docking_scores(output_sdf, manifest, engine="gnina")

        return {
            "docked_poses": output_sdf,
            "scores_csv": scores_csv,
            "log": result.get("log_file", str(work_dir / "gnina.log")),
            "engine": "gnina",
        }

    # ─── DiffDock PPI靶点对接 ────────────────────────────────────────

    def _run_diffdock(self, step: WorkflowStep, inputs: dict[str, Any],
                      work_dir: Path) -> dict[str, Any]:
        """DiffDock对接 (PPI靶点推荐)."""
        from autovs.docking import submit_diffdock_docking

        receptor = ensure_within(
            inputs.get("receptor_pdb", inputs.get("protein_path")),
            [self.settings.task_root], must_exist=True,
        )

        # DiffDock处理单个配体，对于批量需要逐个提交
        # 获取配体SMILES (从manifest或ligands_sdf)
        ligands = inputs.get("ligands_sdf", "")
        if not ligands:
            raise ValueError("DiffDock requires ligands_sdf input")

        # 从配置读取GPU资源
        gpu_config = dict(self.settings.raw.get("slurm", {}).get("gpu", {}))
        gpu_config["gres"] = "gpu:a100_2g.20gb:1"  # DiffDock needs a GPU
        gpu_config["memory"] = "40G"

        result = submit_diffdock_docking(
            receptor_pdb=Path(str(receptor)),
            ligands_smi=str(ligands),
            output_dir=work_dir,
            samples_per_complex=step.parameters.get("samples_per_complex", 10),
            inference_steps=step.parameters.get("inference_steps", 20),
            conda_env="diffdock",
            diffdock_home="/users_home/wangpengzheng/software/DiffDock",
            gpu_config=gpu_config,
            submit_slurm=step.parameters.get("submit_slurm", True),
        )

        slurm_id = result.get("slurm_job_id", "")
        if slurm_id:
            from autovs.af3 import ToolPending
            state_path = work_dir / "diffdock_state.json"
            state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            raise ToolPending(
                f"DiffDock Slurm job {slurm_id} submitted",
                state_path=state_path,
                slurm_job_id=slurm_id,
            )

        # 本地模式: 读取result.json
        result_json_path = Path(result.get("result_json", ""))
        if result_json_path.is_file():
            diffdock_result = json.loads(result_json_path.read_text(encoding="utf-8"))
            poses_count = len(diffdock_result.get("poses", []))
            return {
                "result_json": result_json_path,
                "poses_count": poses_count,
                "top_confidence": diffdock_result.get("top_confidence"),
                "engine": "diffdock",
            }

        raise RuntimeError("DiffDock result.json not found")

    # ─── 多样性选择 ──────────────────────────────────────────────────

    def _run_diversity(self, inputs: dict[str, Any],
                       work_dir: Path) -> dict[str, Any]:
        """基于Murcko骨架的Top-N多样性选择."""
        from autovs.docking import select_diverse_top_n

        scores_csv = ensure_within(
            inputs["scores_csv"], [self.settings.task_root], must_exist=True,
        )
        output_csv = work_dir / "diverse_top20.csv"
        manifest = inputs.get("manifest_csv")

        top_n = int(self.settings.limit("final_hits", 20))
        max_per = int(inputs.get("max_per_scaffold", 2))

        rows = select_diverse_top_n(
            scores_csv=Path(str(scores_csv)),
            output_csv=output_csv,
            top_n=top_n,
            max_per_scaffold=max_per,
            manifest_csv=Path(str(manifest)) if manifest else None,
        )

        return {
            "top_hits": output_csv,
            "hit_count": len(rows),
            "max_per_scaffold": max_per,
        }

    # ─── smina CPU 对接 (legacy) ──────────────────────────────────────
        from rdkit import Chem
        smina_cfg = self._require_executor("smina")
        smina = Path(smina_cfg.path) if smina_cfg.path else None
        if not smina or not smina.exists():
            raise RuntimeError("smina 二进制不可用：请确认 config/tools.toml [executors.smina]")
        receptor = ensure_within(inputs["receptor_pdbqt"], [self.settings.task_root], must_exist=True)
        ligands = ensure_within(inputs["ligands_sdf"], [self.settings.task_root], must_exist=True)
        center, size = inputs.get("center"), inputs.get("size", [24, 24, 24])
        if not center or len(center) != 3:
            raise ValueError("docking requires a three-value pocket center")
        output, log = work_dir / "smina_poses.sdf", work_dir / "smina.log"
        argv = [str(smina), "-r", str(receptor), "-l", str(ligands),
                "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
                "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
                "--exhaustiveness", str(step.parameters.get("exhaustiveness", 4)),
                "--num_modes", str(step.parameters.get("num_modes", 3)),
                "--cpu", str(step.resource_profile.cpus), "--out", str(output)]
        result = run_argv(argv, cwd=work_dir, timeout=step.resource_profile.timeout_seconds, log_path=log)
        if result.returncode or not output.exists():
            raise RuntimeError(f"smina failed: {result.stderr[-500:]}")
        manifest_rows = {}
        manifest_path = inputs.get("manifest_csv")
        if manifest_path:
            checked_manifest = ensure_within(manifest_path, [self.settings.task_root], must_exist=True)
            with checked_manifest.open(encoding="utf-8-sig", newline="") as handle:
                manifest_rows = {row["source_id"]: row for row in csv.DictReader(handle)}
        scores_path = work_dir / "smina_scores.csv"
        score_rows = []
        best: dict[str, dict] = {}
        supplier = Chem.SDMolSupplier(str(output), removeHs=False, strictParsing=False)
        for mol in supplier:
            if mol is None:
                continue
            try:
                source_id = mol.GetProp("source_id") if mol.HasProp("source_id") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            except (RuntimeError, KeyError):
                continue
            affinity = None
            for prop in ("minimizedAffinity", "affinity", "SCORE"):
                if mol.HasProp(prop):
                    try:
                        affinity = float(mol.GetProp(prop)); break
                    except ValueError:
                        pass
            if affinity is None:
                continue
            row = dict(manifest_rows.get(source_id, {}))
            smiles = row.get("smiles", "")
            if not smiles:
                try:
                    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                except Exception:
                    smiles = ""
            row.update({"source_id": source_id, "smiles": smiles,
                        "docking_affinity": affinity})
            if source_id not in best or affinity < float(best[source_id]["docking_affinity"]):
                best[source_id] = row
        score_rows = list(best.values())
        if not score_rows:
            raise RuntimeError("smina output contains no parseable molecule scores")
        fields = list(dict.fromkeys(key for row in score_rows for key in row))
        with scores_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(score_rows)
        return {"docked_poses": output, "scores_csv": scores_path, "log": log}

    def _extract_poses(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        from rdkit import Chem
        receptor = ensure_within(inputs["receptor_pdb"], [self.settings.task_root], must_exist=True)
        docked = ensure_within(inputs["docked_poses"], [self.settings.task_root], must_exist=True)
        engine = str(inputs.get("engine", "smina"))
        metric = str(inputs.get("pose_metric", "best_cnn_vs" if engine == "gnina" else "best_affinity"))
        best: dict[str, tuple[float, Any]] = {}
        for mol in Chem.SDMolSupplier(str(docked), removeHs=False):
            if mol is None:
                continue
            source_id = mol.GetProp("source_id") if mol.HasProp("source_id") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            try:
                if metric == "best_cnn_vs":
                    value = float(mol.GetProp("CNNscore")) * float(mol.GetProp("CNNaffinity")); better = lambda new, old: new > old
                else:
                    for prop in ("minimizedAffinity", "affinity", "SCORE"):
                        if mol.HasProp(prop):
                            value = float(mol.GetProp(prop)); break
                    else:
                        continue
                    better = lambda new, old: new < old
            except (ValueError, KeyError):
                continue
            if source_id not in best or better(value, best[source_id][0]):
                best[source_id] = (value, Chem.Mol(mol))
        if not best:
            raise RuntimeError("no representative docking poses could be selected")
        selected_sdf = work_dir / "selected_poses.sdf"
        writer = Chem.SDWriter(str(selected_sdf))
        receptor_text = receptor.read_text(errors="ignore")
        receptor_body = "\n".join(line for line in receptor_text.splitlines() if not line.startswith("END")) + "\n"
        complexes = work_dir / "complexes"; complexes.mkdir(exist_ok=True)
        index_path = work_dir / "complex_index.csv"
        rows = []
        for source_id, (metric_value, mol) in sorted(best.items()):
            writer.write(mol)
            ligand_block = Chem.MolToPDBBlock(mol)
            complex_path = complexes / f"{source_id}.pdb"
            complex_path.write_text(receptor_body + ligand_block, encoding="utf-8")
            rows.append({"source_id": source_id, "complex_pdb": str(complex_path), "pose_metric": metric, "pose_metric_value": metric_value})
        writer.close()
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            out = csv.DictWriter(handle, fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)
        return {"selected_poses": selected_sdf, "complex_index": index_path, "complex_count": len(rows)}

    def _run_plip(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        plip_cfg = self._require_executor("plip")
        plip = Path(plip_cfg.path) if plip_cfg.path else None
        if not plip or not plip.exists():
            raise RuntimeError("PLIP 二进制不可用：请确认 config/tools.toml [executors.plip]")
        index = ensure_within(inputs["complex_index"], [self.settings.task_root], must_exist=True)
        key_residues = {_normalize_residue(x) for x in inputs.get("key_residues", [])}
        output_root = work_dir / "raw"; output_root.mkdir(exist_ok=True)
        summaries, failures = [], []
        with index.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            source_id = row["source_id"]
            complex_path = ensure_within(row["complex_pdb"], [self.settings.task_root], must_exist=True)
            out_dir = output_root / source_id; out_dir.mkdir(exist_ok=True)
            log = out_dir / "plip.runner.log"
            result = run_argv([str(plip), "-f", str(complex_path), "-o", str(out_dir), "-x", "-t", "--maxthreads", "1"],
                              cwd=out_dir, timeout=1800, log_path=log)
            report_xml = out_dir / "report.xml"
            if not report_xml.exists():
                matches = sorted(out_dir.glob("*_report.xml"))
                report_xml = matches[0] if matches else report_xml
            if result.returncode or not report_xml.exists():
                failures.append({"source_id": source_id, "reason": result.stderr[-500:] or "report.xml missing"})
                continue
            summary = _score_plip_xml(report_xml, key_residues)
            report_txt = out_dir / "report.txt"
            if not report_txt.exists():
                text_matches = sorted(out_dir.glob("*_report.txt"))
                report_txt = text_matches[0] if text_matches else report_txt
            summary.update({"source_id": source_id, "report_xml": str(report_xml), "report_txt": str(report_txt)})
            summaries.append(summary)
        if not summaries:
            raise RuntimeError(f"PLIP produced no successful reports; failures={len(failures)}")
        scores = work_dir / "plip_scores.csv"; failed = work_dir / "failed.csv"
        with scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
        with failed.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_id", "reason"]); writer.writeheader(); writer.writerows(failures)
        return {"plip_scores": scores, "failed": failed, "success_count": len(summaries), "failed_count": len(failures)}

    def _analyze_structure(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Gemmi 结构解析：验证 PDB/mmCIF，提取配体、链和口袋残基信息。"""
        from autovs.gemmi_utils import (
            find_ligands, find_residues_around, read_structure,
            structure_summary, validate_structure,
        )

        protein_path = inputs.get("protein_path")
        if not protein_path:
            raise ValueError("STRUCTURE_ANALYSIS 需要 protein_path 输入")
        protein = ensure_within(protein_path, self.allowed_roots, must_exist=True)

        # 结构验证
        validation = validate_structure(protein)
        validation_path = work_dir / "structure_validation.json"
        validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

        # 读取结构并提取配体
        structure = read_structure(protein)
        ligands = find_ligands(structure)

        # 口袋中心残基搜索
        pocket_residues: list[dict[str, Any]] = []
        center = inputs.get("center")
        if center and len(center) == 3:
            radius = float(inputs.get("radius", 8.0))
            pocket_residues = find_residues_around(
                structure,
                (float(center[0]), float(center[1]), float(center[2])),
                radius=radius,
            )

        # 结构摘要
        summary = structure_summary(structure)
        summary_path = work_dir / "structure_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

        output = {
            "structure_validation": str(validation_path),
            "structure_summary": str(summary_path),
            "atom_count": validation["atom_count"],
            "chain_count": len(validation["chains"]),
            "ligand_count": len(ligands),
            "ligands": ligands,
            "pocket_residues": pocket_residues,
            "resolution": validation.get("resolution"),
            "issues": validation.get("issues", []),
        }

        # 配体详细信息写入 JSON
        if ligands:
            lig_path = work_dir / "ligands.json"
            lig_path.write_text(json.dumps(ligands, ensure_ascii=False, indent=2), encoding="utf-8")
            output["ligands_json"] = str(lig_path)

        if pocket_residues:
            pocket_path = work_dir / "pocket_residues.json"
            pocket_path.write_text(json.dumps(pocket_residues, ensure_ascii=False, indent=2), encoding="utf-8")
            output["pocket_residues_json"] = str(pocket_path)

        return output

    def _repair_protein(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """PDBFixer 蛋白修复：补缺失原子/氢、处理非标准残基、删链/异质分子。"""
        from autovs.pdbfixer_utils import repair_structure, quick_diagnostic

        protein_path = inputs.get("protein_path")
        if not protein_path:
            raise ValueError("PROTEIN_REPAIR 需要 protein_path 输入")
        protein = ensure_within(protein_path, self.allowed_roots, must_exist=True)

        output_pdb = work_dir / "repaired.pdb"

        # 先做快速诊断
        diagnostic = quick_diagnostic(protein)
        diagnostic_path = work_dir / "pre_repair_diagnostic.json"
        diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")

        # 执行修复
        report = repair_structure(
            protein,
            output_pdb,
            add_hydrogens=inputs.get("add_hydrogens", True),
            add_missing_atoms=inputs.get("add_missing_atoms", True),
            replace_nonstandard=inputs.get("replace_nonstandard", True),
            remove_heterogens=inputs.get("remove_heterogens", True),
            keep_chains=inputs.get("keep_chains"),
            remove_chains=inputs.get("remove_chains"),
            ph=float(inputs.get("ph", 7.4)),
            long_gap_threshold=int(inputs.get("long_gap_threshold", 5)),
        )

        report_path = work_dir / "repair_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        return {
            "repaired_structure": str(output_pdb),
            "pre_repair_diagnostic": str(diagnostic_path),
            "repair_report": str(report_path),
            "long_gap_count": len(report["long_gaps"]),
            "warnings": report["warnings"],
            "missing_atoms_fixed": report["missing_atoms_summary"].get("total_missing", 0),
        }

    def _protonate(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """PDB2PQR + PROPKA 质子化：pH 可配的加氢与电荷处理。"""
        from autovs.protonation_utils import protonate_structure, predict_pka

        protein_path = inputs.get("protein_path")
        if not protein_path:
            raise ValueError("PROTONATION 需要 protein_path 输入")
        protein = ensure_within(protein_path, self.allowed_roots, must_exist=True)

        ph = float(inputs.get("ph", 7.4))
        forcefield = str(inputs.get("forcefield", "PARSE"))

        # 1. 先跑 PROPKA pKa 预测
        pka_output = work_dir / "propka_output.csv"
        try:
            pka_report = predict_pka(protein, ph=ph, output_path=pka_output)
        except Exception as exc:
            pka_report = {"error": str(exc), "residues": []}
        pka_path = work_dir / "pka_prediction.json"
        pka_path.write_text(json.dumps(pka_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        # 2. 执行 PDB2PQR 质子化
        output_pqr = work_dir / "output.pqr"
        pdb_out = work_dir / "protonated.pdb"
        report = protonate_structure(
            protein,
            output_pqr,
            ph=ph,
            forcefield=forcefield,
            drop_water=inputs.get("drop_water", True),
            nodebump=inputs.get("nodebump", False),
            noopt=inputs.get("noopt", False),
            pdb_output=pdb_out,
            chains=inputs.get("chains"),
        )

        report_path = work_dir / "protonation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3. 电荷摘要
        from autovs.protonation_utils import quick_charge_summary
        charge_summary = quick_charge_summary(output_pqr)
        charge_path = work_dir / "charge_summary.json"
        charge_path.write_text(json.dumps(charge_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "protonated_pdb": str(pdb_out),
            "output_pqr": str(output_pqr),
            "pka_prediction": str(pka_path),
            "protonation_report": str(report_path),
            "charge_summary": str(charge_path),
            "total_charge": charge_summary.get("total_charge", 0.0),
            "ph": ph,
            "forcefield": forcefield,
            "warnings": report.get("warnings", []),
        }

    def _standardize_v2(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """ChEMBL Structure Pipeline 标准化+去盐。"""
        from autovs.molecule_prep import standardize_molecules

        library_path = inputs.get("library_path")
        if not library_path:
            raise ValueError("需要 library_path")
        library = ensure_within(library_path, self.allowed_roots, must_exist=True)
        output = work_dir / "standardized.smi"
        report = standardize_molecules(
            library, output,
            remove_salts=inputs.get("remove_salts", True),
            neutralize=inputs.get("neutralize", False),
        )
        report_path = work_dir / "standardization_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "standardized_library": output,
            "standardization_report": report_path,
            "success": report["success"],
            "failed": report["failed"],
        }

    def _enumerate_3d(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Gypsum-DL 3D-ready 枚举。"""
        from autovs.molecule_prep import prepare_ligands_3d

        library_path = inputs.get("library_path")
        if not library_path:
            raise ValueError("需要 library_path")
        library = ensure_within(library_path, self.allowed_roots, must_exist=True)
        output = work_dir / "ligands_3d.sdf"
        report = prepare_ligands_3d(
            library, output,
            ph=float(inputs.get("ph", 7.4)),
            max_variants_per_compound=int(inputs.get("max_variants", 4)),
            max_conformers=int(inputs.get("max_conformers", 3)),
        )
        report_path = work_dir / "ligand_3d_enumeration_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"prepared_3d_sdf": output if output.is_file() else None,
                "enumeration_report": report_path,
                "variant_count": report.get("variant_count", 0),
                "returncode": report.get("returncode", 0)}

    def _enumerate_ionization(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Dimorphite-DL pH 依赖离子化枚举。"""
        from autovs.molecule_prep import enumerate_ionization

        smiles_list = inputs.get("smiles_list", [])
        if not smiles_list:
            # 从文件中读取 SMILES
            lib = inputs.get("library_path", "")
            if lib:
                lib_path = ensure_within(lib, self.allowed_roots, must_exist=True)
                with lib_path.open(encoding="utf-8") as f:
                    smiles_list = [line.strip() for line in f if line.strip() and "\t" in line]
        states = enumerate_ionization(
            smiles_list,
            ph_min=float(inputs.get("ph_min", 7.4)),
            ph_max=float(inputs.get("ph_max", 7.4)),
            max_states=int(inputs.get("max_states", 4)),
        )
        ionized = work_dir / "ionized.smi"
        with ionized.open("w", encoding="utf-8") as handle:
            for item in states:
                handle.write(f"{item['source_id']}__ion{item['variant_index']}\t{item['smiles']}\n")
        report_path = work_dir / "ionization_report.json"
        report_path.write_text(json.dumps({
            "count": len(states),
            "ph_min": float(inputs.get("ph_min", 7.4)),
            "ph_max": float(inputs.get("ph_max", 7.4)),
            "max_states": int(inputs.get("max_states", 4)),
            "states_preview": states[:100],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ionized_library": ionized, "ionization_report": report_path,
                "ionization_states": states[:100], "count": len(states)}

    def _prepare_pdbqt(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Meeko PDBQT 参数化。"""
        from autovs.molecule_prep import prepare_pdbqt

        library_path = inputs.get("library_path")
        if not library_path:
            raise ValueError("需要 library_path（SDF 格式）")
        library = ensure_within(library_path, self.allowed_roots, must_exist=True)
        output = work_dir / "ligands.pdbqt"
        report = prepare_pdbqt(
            library, output,
            ph=float(inputs.get("ph", 7.4)),
        )
        report_path = work_dir / "pdbqt_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"prepared_pdbqt": output if report["success"] > 0 else None,
                "pdbqt_report": report_path,
                "success_count": report["success"], "total": report["total"]}

    def _convert_format(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Open Babel 格式转换。"""
        from autovs.molecule_prep import obabel_convert

        library_path = inputs.get("library_path")
        if not library_path:
            raise ValueError("需要 library_path")
        library = ensure_within(library_path, self.allowed_roots, must_exist=True)
        out_format = str(inputs.get("output_format", "sdf"))
        output = work_dir / f"converted.{out_format}"
        obabel_cfg = self._require_executor("obabel")
        report = obabel_convert(
            library, output,
            input_format=str(inputs.get("input_format", "smi")),
            output_format=out_format,
            gen3d=inputs.get("gen3d", False),
            add_hydrogens=inputs.get("add_hydrogens", True),
            ph=float(inputs.get("ph", 7.4)),
            obabel_path=obabel_cfg.path,
        )
        report_path = work_dir / "conversion_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"converted": output, "conversion_report": report_path,
                "molecules": report.get("molecules", 0)}


    def _run_admet(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """ADMET-AI v2.0.1 预测：通过 conda run 调用 wrapper 脚本。"""
        import shutil

        scores_csv = ensure_within(inputs["scores_csv"], [self.settings.task_root], must_exist=True)
        output_csv = work_dir / "admet_predictions.csv"

        # 如果输出已存在且非空，直接复用（幂等性）
        if output_csv.is_file() and output_csv.stat().st_size > 0:
            return {"admet_predictions": output_csv, "molecule_count": -1}

        # 找到 autovs-admet conda 环境中的 Python
        conda_bin = self.settings.executable("conda")
        if not conda_bin or not Path(str(conda_bin)).exists():
            raise RuntimeError("conda 不可用：无法找到 conda 二进制")

        env_python = Path(str(conda_bin)).parent.parent / "envs" / "autovs-admet" / "bin" / "python"
        if not env_python.exists():
            raise RuntimeError(
                "autovs-admet conda 环境未安装。请运行: "
                "conda create -n autovs-admet python=3.12 -y && "
                "conda run -n autovs-admet pip install -e admet_ai/"
            )

        # 找到 wrapper 脚本
        wrapper = Path(self.settings.config_path).parent.parent / "scripts" / "admet_wrapper.py"
        if not wrapper.is_file():
            raise RuntimeError(f"ADMET wrapper 脚本不存在: {wrapper}")

        # 执行预测
        result = run_argv(
            [str(env_python), str(wrapper), str(scores_csv), str(output_csv)],
            cwd=work_dir,
            timeout=int(inputs.get("timeout_seconds", 7200)),
            log_path=work_dir / "admet.log",
        )
        if result.returncode != 0 or not output_csv.is_file():
            raise RuntimeError(
                f"ADMET-AI 预测失败 (exit={result.returncode}): {result.stderr[-500:]}"
            )

        # 读取输出统计分子数
        molecule_count = 0
        try:
            import csv as _csv
            with output_csv.open(encoding="utf-8-sig", newline="") as handle:
                molecule_count = sum(1 for _ in _csv.DictReader(handle))
        except Exception:
            pass

        return {
            "admet_predictions": output_csv,
            "molecule_count": molecule_count,
            "admet_log": work_dir / "admet.log",
        }


    def _run_pose_validation(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """PoseBusters 姿势合理性验证（化学/分子内/分子间检查）。"""
        from posebusters import PoseBusters

        selected_poses = ensure_within(inputs["selected_poses"], [self.settings.task_root], must_exist=True)
        receptor = ensure_within(inputs.get("receptor_pdb") or inputs.get("protein_path"),
                                 [self.settings.task_root], must_exist=True)
        config = str(inputs.get("pb_config", "dock"))
        top_n = inputs.get("top_n")  # None = all
        full_report = bool(inputs.get("full_report", True))

        # 幂等检查
        output_csv = work_dir / "posebusters_report.csv"
        if output_csv.is_file() and output_csv.stat().st_size > 0:
            import csv as _csv
            valid_count = sum(1 for _ in _csv.DictReader(output_csv.open(encoding="utf-8-sig", newline="")))
            return {"pose_validation_report": output_csv, "total_checked": valid_count}

        # 运行 PoseBusters
        busters = PoseBusters(config=config, top_n=top_n)
        result_df = busters.bust(
            mol_pred=str(selected_poses),
            mol_cond=str(receptor),
            full_report=full_report,
        )

        # 写入结果 CSV
        result_df.to_csv(output_csv, index=True, index_label="mol_id")
        log_path = work_dir / "posebusters.log"
        log_path.write_text(
            f"PoseBusters config={config}, molecules={len(result_df)}\n"
            f"columns={list(result_df.columns)}\n",
            encoding="utf-8",
        )

        # 统计
        pb_valid_col = "valid" if "valid" in result_df.columns else None
        total = len(result_df)
        valid_count = int(result_df[pb_valid_col].sum()) if pb_valid_col else total

        return {
            "pose_validation_report": output_csv,
            "pose_validation_log": log_path,
            "total_checked": total,
            "pb_valid_count": valid_count,
            "pb_invalid_count": total - valid_count,
        }


    def _run_pocket_prediction(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """P2Rank ML-based apo pocket prediction.

        Outputs a PocketResolution-compatible JSON with top-N predicted pockets.
        """
        import csv as _csv

        protein = ensure_within(inputs["protein_path"], self.allowed_roots, must_exist=True)
        top_n = int(inputs.get("top_n", 5))
        config_preset = str(inputs.get("p2rank_config", "default"))

        # 幂等检查
        output_json = work_dir / "p2rank_pockets.json"
        if output_json.is_file() and output_json.stat().st_size > 0:
            return {"pocket_resolution": output_json}

        # 查找 P2Rank 可执行脚本
        p2rank_cfg = self._require_executor("p2rank")
        prank_path = Path(p2rank_cfg.path) if p2rank_cfg.path else None
        if not prank_path or not prank_path.exists():
            raise RuntimeError("P2Rank prank 脚本不存在")

        # 设置 Java 版本（P2Rank 需要 Java 17+）
        import os as _os
        env = dict(_os.environ)
        java_home = str(self.settings.executable("conda") or "")
        if java_home:
            java_home = str(Path(java_home).parent.parent)
            env["JAVA_HOME"] = java_home

        # 运行 P2Rank
        argv = [str(prank_path), "predict", "-f", str(protein), "-o", str(work_dir)]
        if config_preset != "default":
            argv.extend(["-c", config_preset])
        log = work_dir / "p2rank.log"
        result = run_argv(
            argv, cwd=work_dir, timeout=int(inputs.get("timeout_seconds", 3600)),
            log_path=log, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"P2Rank failed (exit={result.returncode}): {result.stderr[-500:]}")

        # 查找预测输出 CSV
        predictions_csv = next(work_dir.glob("*_predictions.csv"), None)
        if not predictions_csv:
            raise RuntimeError("P2Rank did not produce a _predictions.csv file")

        # 解析口袋候选
        pockets: list[dict] = []
        with predictions_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = _csv.DictReader(handle)
            for row in reader:
                try:
                    name = (row.get("name") or "").strip()
                    rank = int(float((row.get("rank") or "0").strip()))
                    score = float((row.get("score") or "0").strip())
                    prob = float((row.get("probability") or "0").strip())
                    cx = float((row.get("center_x") or "0").strip())
                    cy = float((row.get("center_y") or "0").strip())
                    cz = float((row.get("center_z") or "0").strip())
                    residue_str = (row.get("residue_ids") or "").strip()
                    residues = [r.strip() for r in residue_str.split() if r.strip()] if residue_str else []
                except (ValueError, KeyError):
                    continue
                pockets.append({
                    "name": name, "rank": rank, "score": round(score, 2),
                    "probability": round(prob, 3),
                    "center": [round(cx, 4), round(cy, 4), round(cz, 4)],
                    "residue_ids": residues,
                })
                if len(pockets) >= top_n:
                    break

        if not pockets:
            raise RuntimeError("P2Rank produced no valid pocket predictions")

        # 构建 PocketResolution 兼容输出
        from autovs.schemas import (
            PocketCandidate, PocketConfidence, PocketEvidence,
            PocketQualityGate, PocketResolution, PocketSource,
        )
        import hashlib as _hashlib
        top = pockets[0]
        selected = PocketCandidate(
            pocket_id=f"pocket_{_hashlib.sha256(str(top).encode()).hexdigest()[:12]}",
            rank=1,
            center=tuple(top["center"]),
            size=(24.0, 24.0, 24.0),
            source=PocketSource.VERIFIED_RESEARCH_STRUCTURE,
            confidence=PocketConfidence.MEDIUM,
            residues=top["residue_ids"],
            evidence=[
                PocketEvidence(kind="p2rank_score",
                               description=f"P2Rank score: {top['score']}, prob: {top['probability']}",
                               value=top["score"]),
                PocketEvidence(kind="p2rank_config",
                               description=f"P2Rank config: {config_preset}",
                               value=config_preset),
            ],
            quality_gates=[
                PocketQualityGate(name="p2rank_prediction", status="passed",
                                  detail=f"Top-1 score={top['score']} among {len(pockets)} candidates"),
            ],
            tool_versions={"p2rank": "2.5.1"},
        )
        alternates = []
        for p in pockets[1:top_n]:
            alternates.append(PocketCandidate(
                pocket_id=f"pocket_{_hashlib.sha256(str(p).encode()).hexdigest()[:12]}",
                rank=p["rank"],
                center=tuple(p["center"]),
                size=(24.0, 24.0, 24.0),
                source=PocketSource.VERIFIED_RESEARCH_STRUCTURE,
                confidence=PocketConfidence.MEDIUM,
                residues=p["residue_ids"],
                evidence=[
                    PocketEvidence(kind="p2rank_score",
                                   description=f"P2Rank score: {p['score']}, prob: {p['probability']}",
                                   value=p["score"]),
                ],
            ))
        resolution = PocketResolution(
            protein_path=str(protein),
            selected_pocket=selected,
            alternate_pockets=alternates,
            research_pdb_id=Path(protein).stem,
            warnings=[f"P2Rank ML prediction (config={config_preset}); verify with experimental evidence if available"],
        )
        output_json.write_text(resolution.model_dump_json(indent=2), encoding="utf-8")
        return {"pocket_resolution": output_json, "pocket": output_json,
                "predicted_pockets_csv": predictions_csv,
                "total_pockets": len(pockets), "top_score": top["score"]}


    def _run_diffdock(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """DiffDock: 扩散模型分子对接（conda env 调用 wrapper）。"""
        receptor = ensure_within(inputs["receptor_pdb"], [self.settings.task_root], must_exist=True)
        ligand_desc = inputs.get("ligand_smiles") or inputs.get("ligand_sdf")
        if not ligand_desc:
            raise ValueError("DIFFDOCK_DOCKING 需要 ligand_smiles 或 ligand_sdf 输入")
        samples = int(inputs.get("samples", 10))
        steps = int(inputs.get("inference_steps", 20))

        # 幂等
        output_json = work_dir / "diffdock_result.json"
        if output_json.is_file():
            import json as _json
            data = _json.loads(output_json.read_text(encoding="utf-8"))
            if data.get("status") == "ok":
                return {"diffdock_result": output_json, "docked_poses": work_dir / "rank1.sdf"}

        # 找 conda env Python 和 wrapper 脚本
        conda_bin = self.settings.executable("conda")
        env_python = Path(str(conda_bin)).parent.parent / "envs" / "diffdock" / "bin" / "python"
        if not env_python.exists():
            raise RuntimeError("diffdock conda 环境未安装")

        wrapper = Path(str(self.settings.config_path)).parent.parent / "scripts" / "diffdock_wrapper.py"
        if not wrapper.is_file():
            raise RuntimeError(f"DiffDock wrapper 不存在: {wrapper}")

        # 如果 ligand_desc 是 SDF 路径，确保它在 task_root 内
        if Path(ligand_desc).exists():
            ligand_desc = str(ensure_within(ligand_desc, [self.settings.task_root], must_exist=True))

        argv = [
            str(env_python), str(wrapper),
            str(receptor), str(ligand_desc), str(work_dir),
            "--samples", str(samples),
            "--steps", str(steps),
        ]
        result = run_argv(
            argv, cwd=Path("/users_home/wangpengzheng/software/DiffDock"),  # SO(2) 缓存在此目录
            timeout=int(inputs.get("timeout_seconds", 36000)),
            log_path=work_dir / "diffdock.log",
        )
        if result.returncode != 0 or not output_json.is_file():
            raise RuntimeError(f"DiffDock failed (exit={result.returncode}): {result.stderr[-500:]}")

        # 解析结果找最佳姿态
        data = json.loads(output_json.read_text(encoding="utf-8"))
        poses = data.get("poses", [])
        top_confidence = data.get("top_confidence")
        rank1_sdf = work_dir / "rank1.sdf"

        return {
            "diffdock_result": output_json,
            "docked_poses": rank1_sdf if rank1_sdf.is_file() else None,
            "top_confidence": top_confidence,
            "pose_count": len(poses),
        }


    def _run_fpocket(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """fpocket: 几何口袋检测（Voronoi alpha sphere + 描述符）。"""
        import numpy as np
        import shutil

        protein = ensure_within(inputs["protein_path"], self.allowed_roots, must_exist=True)
        top_n = int(inputs.get("top_n", 5))

        output_json = work_dir / "fpocket_pockets.json"
        # 幂等
        if output_json.is_file() and output_json.stat().st_size > 0:
            return {"pocket_resolution": output_json}

        fpocket_cfg = self._require_executor("fpocket")
        fpocket_bin = Path(fpocket_cfg.path) if fpocket_cfg.path else None
        if not fpocket_bin or not fpocket_bin.exists():
            raise RuntimeError("fpocket 二进制不存在")

        # 复制蛋白到工作目录（fpocket 输出在 PDB 同目录）
        local_pdb = work_dir / protein.name
        shutil.copy2(protein, local_pdb)

        result = run_argv(
            [str(fpocket_bin), "-f", str(local_pdb)],
            cwd=work_dir, timeout=int(inputs.get("timeout_seconds", 600)),
            log_path=work_dir / "fpocket.log",
        )
        if result.returncode != 0:
            raise RuntimeError(f"fpocket failed (exit={result.returncode}): {result.stderr[-500:]}")

        # 查找输出目录
        stem = local_pdb.stem
        out_dir = work_dir / f"{stem}_out"
        if not out_dir.is_dir():
            raise RuntimeError(f"fpocket 输出目录未找到: {out_dir}")

        # 解析 info.txt
        info_txt = out_dir / f"{stem}_info.txt"
        if not info_txt.is_file():
            raise RuntimeError(f"fpocket info 文件未找到: {info_txt}")

        pockets: list[dict] = []
        current = None
        for line in info_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("Pocket"):
                if current:
                    pockets.append(current)
                parts = line.split(":")
                pid = int(parts[0].split()[-1]) if parts else len(pockets) + 1
                current = {"rank": pid}
            elif current is not None:
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip(), val.strip()
                    try:
                        current[key] = float(val) if "." in val or val.replace("-", "").isdigit() else val
                    except ValueError:
                        current[key] = val
        if current:
            pockets.append(current)

        # 解析口袋 PDB 提取中心坐标
        for p in pockets:
            rank = p["rank"]
            pocket_pdb = out_dir / "pockets" / f"pocket{rank}_atm.pdb"
            if not pocket_pdb.is_file():
                continue
            coords = []
            for pline in pocket_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
                if pline.startswith("ATOM") or pline.startswith("HETATM"):
                    try:
                        coords.append([float(pline[30:38]), float(pline[38:46]), float(pline[46:54])])
                    except (ValueError, IndexError):
                        pass
            if coords:
                arr = np.array(coords)
                p["center"] = [round(float(arr[:, 0].mean()), 4),
                               round(float(arr[:, 1].mean()), 4),
                               round(float(arr[:, 2].mean()), 4)]
                p["atom_count"] = len(coords)

        # 按 druggability score 排序
        pockets.sort(key=lambda p: float(p.get("Druggability Score", p.get("Score", 0))), reverse=True)
        for i, p in enumerate(pockets):
            p["rank"] = i + 1

        if not pockets:
            raise RuntimeError("fpocket 未检测到任何口袋")

        # 构建 PocketResolution
        from autovs.schemas import (
            PocketCandidate, PocketConfidence, PocketEvidence,
            PocketQualityGate, PocketResolution, PocketSource,
        )
        import hashlib as _hashlib
        top = pockets[0]
        selected = PocketCandidate(
            pocket_id=f"pocket_{_hashlib.sha256(str(top.get('center',[0,0,0])).encode()).hexdigest()[:12]}",
            rank=1,
            center=tuple(top.get("center", [0, 0, 0])),
            size=(24.0, 24.0, 24.0),
            source=PocketSource.VERIFIED_RESEARCH_STRUCTURE,
            confidence=PocketConfidence.MEDIUM,
            residues=[],
            evidence=[
                PocketEvidence(kind="fpocket_score",
                               description=f"Druggability: {top.get('Druggability Score','?')}, Score: {top.get('Score','?')}",
                               value=float(top.get("Druggability Score", top.get("Score", 0)))),
                PocketEvidence(kind="fpocket_volume",
                               description=f"Volume: {top.get('Volume','?')} A^3",
                               value=float(top.get("Volume", 0))),
            ],
            quality_gates=[
                PocketQualityGate(name="fpocket_detection", status="passed",
                                  detail=f"Top druggability={top.get('Druggability Score','?')} among {len(pockets)}"),
            ],
            tool_versions={"fpocket": "4.0"},
        )
        alternates = []
        for p in pockets[1:top_n]:
            drugg = float(p.get("Druggability Score", p.get("Score", 0)))
            alternates.append(PocketCandidate(
                pocket_id=f"pocket_{_hashlib.sha256(str(p.get('center',[0,0,0])).encode()).hexdigest()[:12]}",
                rank=p["rank"],
                center=tuple(p.get("center", [0, 0, 0])),
                size=(24.0, 24.0, 24.0),
                source=PocketSource.VERIFIED_RESEARCH_STRUCTURE,
                confidence=PocketConfidence.MEDIUM if drugg > 0.4 else PocketConfidence.LOW,
                residues=[],
                evidence=[
                    PocketEvidence(kind="fpocket_score",
                                   description=f"Druggability: {p.get('Druggability Score','?')}, Score: {p.get('Score','?')}",
                                   value=drugg),
                ],
            ))
        resolution = PocketResolution(
            protein_path=str(local_pdb),
            selected_pocket=selected,
            alternate_pockets=alternates,
            research_pdb_id=stem,
            warnings=["fpocket geometric detection; verify with ML/experimental evidence"],
        )
        output_json.write_text(resolution.model_dump_json(indent=2), encoding="utf-8")
        return {
            "pocket_resolution": output_json,
            "pocket": output_json,
            "total_pockets": len(pockets),
            "top_druggability": top.get("Druggability Score", top.get("Score")),
        }


    def _run_pharmacophore(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Pharmit: 药效团筛选（pharma → dbcreate → dbsearch）。"""
        import csv as _csv

        library_path = ensure_within(
            inputs.get("library_path") or inputs.get("normalized_library"),
            self.allowed_roots, must_exist=True,
        )
        query_ligand_sdf = inputs.get("query_ligand_sdf")
        query_json = inputs.get("pharmacophore_query_json")

        if not query_ligand_sdf and not query_json:
            raise ValueError("PHARMACOPHORE_SCREENING 需要 query_ligand_sdf 或 pharmacophore_query_json")

        pharmit_cfg = self._require_executor("pharmit")
        pharmit = Path(pharmit_cfg.path) if pharmit_cfg.path else None
        if not pharmit or not pharmit.exists():
            raise RuntimeError("pharmit 二进制不存在")

        # 幂等
        hits_sdf = work_dir / "pharmit_hits.sdf"
        if hits_sdf.is_file() and hits_sdf.stat().st_size > 0:
            return {"pharmacophore_hits": hits_sdf}

        # 1. 生成药效团 query
        if query_json:
            query_file = work_dir / "query.json"
            if isinstance(query_json, dict):
                query_file.write_text(json.dumps(query_json), encoding="utf-8")
            else:
                import shutil
                qp = ensure_within(str(query_json), [self.settings.task_root], must_exist=True)
                shutil.copy2(qp, query_file)
        elif query_ligand_sdf:
            ql = ensure_within(str(query_ligand_sdf), [self.settings.task_root], must_exist=True)
            query_file = work_dir / "pharma_output.json"
            result = run_argv(
                [str(pharmit), "pharma", "-in", str(ql), "-out", str(query_file)],
                cwd=work_dir, timeout=300, log_path=work_dir / "pharma.log",
            )
            if result.returncode != 0:
                raise RuntimeError(f"pharma 失败: {result.stderr[-500:]}")
            # pharma 可能输出多个JSON对象，提取第一个
            raw = query_file.read_text(encoding="utf-8").strip()
            if raw.startswith("{"):
                pass  # 单对象OK
            elif "}{" in raw:
                first = raw[:raw.index("}{") + 1]
                query_file.write_text(first, encoding="utf-8")
        else:
            raise RuntimeError("无法获取药效团query")

        # 2. 建库 (dbcreate)
        db_dir = work_dir / "pharmit_db"
        db_dir.mkdir(exist_ok=True)
        result = run_argv(
            [str(pharmit), "dbcreate", "-in", str(library_path), "-dbdir", str(db_dir)],
            cwd=work_dir, timeout=int(inputs.get("timeout_seconds", 7200)),
            log_path=work_dir / "dbcreate.log",
        )
        if result.returncode != 0:
            raise RuntimeError(f"dbcreate 失败: {result.stderr[-500:]}")

        # 3. 搜索 (dbsearch)
        result = run_argv(
            [str(pharmit), "dbsearch", "-dbdir", str(db_dir),
             "-in", str(query_file), "-out", str(hits_sdf),
             "-max-hits", str(inputs.get("max_hits", 1000))],
            cwd=work_dir, timeout=int(inputs.get("timeout_seconds", 7200)),
            log_path=work_dir / "dbsearch.log",
        )
        if result.returncode != 0:
            raise RuntimeError(f"dbsearch 失败: {result.stderr[-500:]}")
        # dbsearch返回0即使0命中，检查输出
        hit_count = 0
        if hits_sdf.is_file():
            hit_count = hits_sdf.read_text(encoding="utf-8", errors="replace").count("$$$$")

        return {
            "pharmacophore_hits": hits_sdf if hits_sdf.is_file() and hit_count > 0 else None,
            "pharmacophore_query": query_file,
            "pharmit_db": db_dir,
            "hit_count": hit_count,
        }


    def _run_foldseek(self, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        """Foldseek: 蛋白质结构同源搜索（3Di+GPU加速）。

        输入：query PDB + target DB（或PDB文件列表用于建库）
        输出：结构命中 m8 table（query,target,fident,alnlen,evalue,bits）
        """
        import csv as _csv

        query_pdb = ensure_within(inputs["query_pdb"], self.allowed_roots, must_exist=True)
        target_db = inputs.get("target_db_dir")
        target_pdbs = inputs.get("target_pdb_list", [])

        foldseek_cfg = self._require_executor("foldseek")
        foldseek = Path(foldseek_cfg.path) if foldseek_cfg.path else None
        if not foldseek or not foldseek.exists():
            raise RuntimeError("foldseek 二进制不存在")

        # 幂等检查
        output_m8 = work_dir / "foldseek_results.m8"
        if output_m8.is_file() and output_m8.stat().st_size > 0:
            hits = list(_csv.reader(output_m8.open(), delimiter="\t"))
            return {"foldseek_results": output_m8, "hit_count": len(hits)}

        # 确定目标数据库
        if not target_db:
            if isinstance(target_pdbs, list) and target_pdbs:
                db_dir = work_dir / "target_db"
                db_dir.mkdir(exist_ok=True)
                db_input = work_dir / "targets.txt"
                db_input.write_text("\n".join(str(p) for p in target_pdbs), encoding="utf-8")
                result = run_argv(
                    [str(foldseek), "createdb", str(db_input), str(db_dir / "target")],
                    cwd=work_dir, timeout=3600, log_path=work_dir / "createdb.log",
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Foldseek createdb failed: {result.stderr[-500:]}")
                target_db = str(db_dir / "target")
            else:
                raise ValueError("Need target_db_dir or target_pdb_list for Foldseek search")

        # 结构搜索
        sensitivity = inputs.get("sensitivity", 7.5)
        max_seqs = int(inputs.get("max_seqs", 1000))
        tmp_dir = work_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        format_opts = "query,target,fident,alnlen,mismatch,qcov,tcov,evalue,bits"

        result = run_argv(
            [str(foldseek), "easy-search", str(query_pdb), target_db,
             str(output_m8), str(tmp_dir),
             "--format-output", format_opts,
             "-s", str(sensitivity),
             "--max-seqs", str(max_seqs)],
            cwd=work_dir, timeout=int(inputs.get("timeout_seconds", 7200)),
            log_path=work_dir / "foldseek.log",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Foldseek search failed: {result.stderr[-500:]}")

        # 解析结果
        hits = []
        if output_m8.is_file():
            with output_m8.open(newline="", encoding="utf-8") as f:
                hits = list(_csv.reader(f, delimiter="\t"))

        return {
            "foldseek_results": output_m8,
            "hit_count": len(hits),
            "top_hit": hits[0] if hits else None,
        }


def _jsonable(outputs: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in outputs.items()}


def _walk_output_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_output_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_output_strings(item)


def _normalize_residue(value: str) -> str:
    return value.upper().replace(":", "").replace(" ", "")


def _score_plip_xml(path: Path, key_residues: set[str]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    score, hbonds, key_hbonds, hydrophobic, salt_bridges = 0, [], [], 0, 0
    for node in root.iter():
        tag = node.tag.split("}")[-1].lower()
        if tag not in {"hydrogen_bond", "hydrophobic_interaction", "salt_bridge", "pi_stack", "pi_cation_interaction", "halogen_bond"}:
            continue
        text = {child.tag.split("}")[-1].lower(): (child.text or "").strip() for child in node}
        residue = _normalize_residue(f"{text.get('restype', '')}{text.get('resnr', '')}{text.get('reschain', '')}")
        residue_no_chain = _normalize_residue(f"{text.get('restype', '')}{text.get('resnr', '')}")
        if tag == "hydrogen_bond":
            hbonds.append(residue)
            if residue in key_residues or residue_no_chain in key_residues:
                score += 3; key_hbonds.append(residue)
            else:
                try:
                    distance = float(text.get("dist_h-a") or text.get("dist_d-a") or 99)
                except ValueError:
                    distance = 99
                if distance <= 3.5:
                    score += 2
        elif tag == "hydrophobic_interaction":
            score += 1; hydrophobic += 1
        elif tag == "salt_bridge":
            score += 3; salt_bridges += 1
        else:
            score += 2
    return {"plip_score": score, "hbond_count": len(hbonds), "key_hbond_count": len(key_hbonds),
            "key_hbond_residues": ";".join(sorted(set(key_hbonds))), "all_hbond_residues": ";".join(sorted(set(hbonds))),
            "hydrophobic_count": hydrophobic, "salt_bridge_count": salt_bridges}
