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
            return self._run_smina(step, inputs, work_dir)
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
        raise RuntimeError(f"{action.value} has no active production adapter; capability is not executable yet")

    def _run_smina(self, step: WorkflowStep, inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
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
        for mol in Chem.SDMolSupplier(str(output), removeHs=False):
            if mol is None:
                continue
            source_id = mol.GetProp("source_id") if mol.HasProp("source_id") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
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
            row.update({"source_id": source_id, "smiles": row.get("smiles", Chem.MolToSmiles(Chem.RemoveHs(mol))),
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
