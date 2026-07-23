from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from autovs.af3 import ToolPending


def gromacs_env_available(settings: Any) -> tuple[bool, str]:
    cfg = settings.executor_config("gromacs")
    if not cfg:
        return False, "gromacs executor is not configured"
    image = cfg.resolved_path(str(settings.config_path.parent)) if cfg.path else None
    if not image or not image.exists():
        return False, f"GROMACS Apptainer image not found: {cfg.path}"
    if not settings.executable("sbatch") or not settings.executable("sbatch").exists():
        return False, "Slurm sbatch not found"
    if not settings.executable("apptainer") or not settings.executable("apptainer").exists():
        return False, "Apptainer binary not found"
    return True, f"GROMACS container available: {image}"


def submit_gromacs_md(
    *,
    receptor_pdb: Path,
    selected_poses: Path,
    work_dir: Path,
    settings: Any,
    parameters: dict[str, Any] | None = None,
    short: bool = True,
) -> dict[str, Any]:
    params = parameters or {}
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "gromacs_state.json"
    report_path = work_dir / "gromacs_report.json"
    state = _read_json(state_path)
    if state.get("status") == "succeeded" and state.get("scores_csv") and Path(state["scores_csv"]).is_file():
        return {
            "scores_csv": Path(state["scores_csv"]),
            "gromacs_state": state_path,
            "gromacs_report": report_path,
        }
    if state.get("status") == "submitted":
        raise ToolPending(
            f"GROMACS MD Slurm job(s) {state.get('slurm_job_id', 'unknown')} are still pending external completion",
            state_path=state_path,
            payload=state,
        )

    ok, reason = gromacs_env_available(settings)
    if not ok:
        raise RuntimeError(reason)

    systems_dir = work_dir / "systems"
    systems_dir.mkdir(exist_ok=True)
    max_ligands = int(params.get("max_ligands", settings.limit("short_md_hits" if short else "long_md_hits", 10 if short else 3)))
    run_ns = float(params.get("simulation_ns", 10.0 if short else 100.0))
    manifest, rows = _build_manifest(
        receptor_pdb=receptor_pdb,
        selected_poses=selected_poses,
        systems_dir=systems_dir,
        manifest_path=work_dir / "gromacs_manifest.csv",
        max_ligands=max_ligands,
        run_ns=run_ns,
    )
    submissions = _submit_rows_with_existing_runner(
        rows=rows,
        settings=settings,
        run_ns=run_ns,
        short=short,
        work_dir=work_dir,
    )
    if not submissions or not any(item.get("job_id") for item in submissions):
        raise RuntimeError("GROMACS submission produced no Slurm job ids")
    state = {
        "status": "submitted",
        "slurm_job_id": ",".join(str(item.get("job_id")) for item in submissions if item.get("job_id")),
        "submissions": submissions,
        "manifest": str(manifest),
        "run_ns": run_ns,
        "short": short,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps({
        "message": "GROMACS MD submitted to Slurm; resume after terminal status and result summarization.",
        **state,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    raise ToolPending(
        f"GROMACS MD submitted as Slurm job(s) {state['slurm_job_id']}; resume after completion",
        state_path=state_path,
        payload=state,
    )


def _build_manifest(*, receptor_pdb: Path, selected_poses: Path, systems_dir: Path,
                    manifest_path: Path, max_ligands: int, run_ns: float) -> tuple[Path, list[dict[str, Any]]]:
    from rdkit import Chem

    rows: list[dict[str, Any]] = []
    supplier = Chem.SDMolSupplier(str(selected_poses), removeHs=False)
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        if len(rows) >= max_ligands:
            break
        source_id = mol.GetProp("source_id") if mol.HasProp("source_id") else (
            mol.GetProp("_Name") if mol.HasProp("_Name") else f"ligand_{idx+1:03d}"
        )
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in source_id)
        ligand_path = systems_dir / f"{safe_id}.sdf"
        writer = Chem.SDWriter(str(ligand_path))
        writer.write(mol)
        writer.close()
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        rows.append({
            "task_index": len(rows),
            "system_id": safe_id,
            "mol_id": safe_id,
            "receptor_pdb": str(receptor_pdb),
            "ligand_sdf": str(ligand_path),
            "seed": 1000 + len(rows),
            "run_ns": run_ns,
            "system_dir": str(systems_dir / safe_id),
            "net_charge": charge,
            "charge_audit_status": "pending_autovs_adapter",
        })
    if not rows:
        raise RuntimeError("selected_poses contains no RDKit-readable ligands for MD")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path, rows


def _submit_rows_with_existing_runner(*, rows: list[dict[str, Any]], settings: Any,
                                      run_ns: float, short: bool, work_dir: Path) -> list[dict[str, Any]]:
    from src.tools.molecular_utils import GromacsMDRunner

    slurm = settings.raw.get("slurm", {}).get("gpu", {})
    submissions = []
    for row in rows:
        result = GromacsMDRunner.prepare_and_submit(
            receptor_pdb=row["receptor_pdb"],
            ligand_sdf=row["ligand_sdf"],
            mol_id=row["mol_id"],
            workdir_base=str(work_dir / "submitted_systems"),
            formal_charge=int(row["net_charge"]),
            simulation_ns=run_ns,
            force_field=str(row.get("force_field", "amber99sb-ildn")),
            water_model=str(row.get("water_model", "tip3p")),
            submit_slurm=True,
            slurm_gres=str(slurm.get("gres", "gpu:a100_2g.20gb:1")),
            slurm_cpus=int(slurm.get("cpus_per_task", slurm.get("cpus", 8))),
            slurm_mem=str(slurm.get("memory", "20G")),
            slurm_walltime="1-12:00:00" if short else str(slurm.get("time", "3-00:00:00")),
        )
        submissions.append(result)
    return submissions


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
