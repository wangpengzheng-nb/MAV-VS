from __future__ import annotations

import shutil
from pathlib import Path

from autovs.config import PROJECT_ROOT, Settings
from autovs.library import verify_default_library
from autovs.schemas import ActionType, ToolCapability


CAPABILITY_DEFINITIONS = {
    ActionType.INPUT_VALIDATION: ("Strict input binder", "Validate an optional PDB and a UTF-8 molecule_id<TAB>SMILES library", "python", ["PDB", "strict_smi_v1"], ["JSON", "SMI", "TSV"], False),
    ActionType.TARGET_STRUCTURE_ACQUISITION: ("Controlled RCSB structure acquisition", "Download only research-verified experimental holo PDB IDs from files.rcsb.org", "python", ["JSON"], ["PDB", "JSON"], False),
    ActionType.TARGET_STRUCTURE_PREDICTION: ("Predicted target structure acquisition", "Predict a target structure through a configured AlphaFold or Boltz adapter", "slurm", ["FASTA", "JSON"], ["PDB", "CIF", "JSON"], True),
    ActionType.PROTEIN_PREPARATION: ("OpenBabel protein preparation", "Remove water, add hydrogens and produce receptor files", "conda", ["PDB"], ["PDB", "PDBQT"], False),
    ActionType.POCKET_DEFINITION: ("Evidence-backed pocket resolver", "Resolve and validate a pocket from a user box, uploaded cocrystal ligand, verified research structure, or mapped key residues", "python", ["PDB", "JSON"], ["JSON"], False),
    ActionType.MOLECULE_STANDARDIZATION: ("RDKit standardization", "Canonicalize and filter strict SMI while preserving the user molecule ID", "python", ["strict_smi_v1"], ["CSV", "SDF"], False),
    ActionType.CONFORMER_GENERATION: ("RDKit ETKDGv3", "Generate explicit-H 3D conformers with MMFF94s/UFF", "python", ["strict_smi_v1"], ["SDF", "CSV"], False),
    ActionType.PHYSICOCHEMICAL_FILTERING: ("RDKit filters", "Apply physicochemical, PAINS and reactive-group gates", "python", ["strict_smi_v1"], ["CSV", "SDF"], False),
    ActionType.DIVERSITY_SELECTION: ("Murcko diversity selector", "Limit scaffold monopolization while preserving rank", "python", ["CSV", "SDF"], ["CSV", "SDF"], False),
    ActionType.MOLECULAR_DOCKING: ("smina/GNINA docking", "CPU smina rough docking or GPU GNINA rough/refinement", "slurm", ["PDB", "PDBQT", "SDF", "JSON"], ["SDF", "CSV"], False),
    ActionType.POSE_EXTRACTION: ("Docking pose extractor", "Select best affinity or best CNN_VS pose per molecule", "python", ["SDF"], ["SDF", "CSV"], False),
    ActionType.INTERACTION_ANALYSIS: ("PLIP", "Compute protein-ligand interaction fingerprints", "conda", ["PDB"], ["XML", "TXT", "CSV"], False),
    ActionType.ADMET_FILTERING: ("ADMET-AI", "Predict ADMET risks and physicochemical properties", "conda", ["CSV"], ["CSV"], False),
    ActionType.SHORT_MD: ("GROMACS 10 ns quality gate", "Charge-audited short MD stability check", "apptainer", ["PDB", "SDF", "CSV"], ["XTC", "CSV", "JSON"], True),
    ActionType.MOLECULAR_DYNAMICS: ("GROMACS 100 ns + MMGBSA", "Charge-audited production MD and 70-100 ns MMGBSA", "apptainer", ["PDB", "SDF", "CSV"], ["XTC", "CSV", "JSON"], True),
    ActionType.FINAL_RANKING: ("Evidence ranker", "Direction-aware normalized ranking with scaffold diversity", "python", ["CSV", "SDF"], ["CSV", "SDF"], False),
    ActionType.REPORT_GENERATION: ("Reproducibility reporter", "Generate Markdown, HTML and artifact manifest", "python", ["JSON", "CSV"], ["MD", "HTML", "JSON"], False),
    ActionType.STRUCTURE_ANALYSIS: ("Gemmi structure analyzer", "Validate PDB/mmCIF, detect ligands, extract chains, search pocket residues", "python", ["PDB", "mmCIF"], ["JSON", "TXT"], False),
    ActionType.PROTEIN_REPAIR: ("PDBFixer protein repair", "Add missing atoms/hydrogens, replace nonstandard residues, remove unwanted chains/heterogens", "python", ["PDB"], ["PDB", "JSON"], False),
    ActionType.PROTONATION: ("PDB2PQR + PROPKA protonation", "pH-dependent pKa prediction, hydrogen addition, and forcefield parameter assignment", "python", ["PDB"], ["PQR", "PDB", "JSON"], False),
}


def _exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def list_capabilities(settings: Settings) -> list[ToolCapability]:
    result: list[ToolCapability] = []
    for action, definition in CAPABILITY_DEFINITIONS.items():
        name, desc, executor, inputs, outputs, gpu = definition
        availability, reason = "available", ""
        if action == ActionType.MOLECULAR_DOCKING:
            smina_cfg = settings.executor_config("smina")
            gnina_cfg = settings.executor_config("gnina")
            has_smina = smina_cfg and smina_cfg.exists(str(PROJECT_ROOT)) if smina_cfg else False
            has_gnina = gnina_cfg and gnina_cfg.exists(str(PROJECT_ROOT)) if gnina_cfg else False
            if not has_smina and not has_gnina:
                availability, reason = "unavailable", "neither smina nor GNINA is configured"
            elif not has_gnina:
                availability, reason = "degraded", "GNINA unavailable; CPU smina only"
        elif action == ActionType.TARGET_STRUCTURE_PREDICTION:
            availability, reason = "unavailable", "AlphaFold/Boltz structure prediction adapter is not configured yet"
        elif action == ActionType.POCKET_DEFINITION:
            plip_cfg = settings.executor_config("plip")
            if not plip_cfg or not plip_cfg.exists(str(PROJECT_ROOT)):
                availability, reason = "degraded", "PLIP unavailable; geometric ligand-contact validation remains available"
        elif action == ActionType.INTERACTION_ANALYSIS:
            plip_cfg = settings.executor_config("plip")
            if not plip_cfg or not plip_cfg.exists(str(PROJECT_ROOT)):
                availability, reason = "unavailable", "PLIP binary not found"
        elif action == ActionType.PROTEIN_PREPARATION:
            obabel_cfg = settings.executor_config("obabel")
            if not obabel_cfg or not obabel_cfg.exists(str(PROJECT_ROOT)):
                availability, reason = "unavailable", "OpenBabel binary not found"
        elif action == ActionType.ADMET_FILTERING:
            admet_cfg = settings.executor_config("admet_ai")
            if admet_cfg and admet_cfg.env:
                conda = settings.executable("conda")
                env_path = conda.parent.parent / "envs" / admet_cfg.env if conda else None
                if not _exists(env_path):
                    availability, reason = "degraded", f"conda 环境 {admet_cfg.env} 未安装"
            else:
                availability, reason = "degraded", "admet_ai 未在 [executors] 中配置"
        elif action in {ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS}:
            gromacs_cfg = settings.executor_config("gromacs")
            if not gromacs_cfg or not gromacs_cfg.exists(str(PROJECT_ROOT)):
                availability, reason = "unavailable", "GROMACS Apptainer image not found"
            elif not _exists(settings.executable("sbatch")):
                availability, reason = "unavailable", "Slurm sbatch not found"
            else:
                availability, reason = "degraded", "GPU execution requires a healthy Slurm GPU partition"
        elif action == ActionType.STRUCTURE_ANALYSIS:
            try:
                import gemmi  # noqa: F401
            except ImportError:
                availability, reason = "unavailable", "gemmi Python package not installed (pip install gemmi)"
        elif action == ActionType.PROTEIN_REPAIR:
            try:
                import pdbfixer  # noqa: F401
            except ImportError:
                availability, reason = "unavailable", "pdbfixer Python package not installed (pip install pdbfixer)"
        elif action == ActionType.PROTONATION:
            try:
                import pdb2pqr  # noqa: F401
            except ImportError:
                availability, reason = "unavailable", "pdb2pqr Python package not installed (pip install pdb2pqr)"
        result.append(ToolCapability(
            action_type=action, name=name, description=desc, availability=availability,
            executor=executor, input_formats=inputs, output_formats=outputs,
            gpu_required=gpu, reason=reason,
        ))
    return result


def health_report(settings: Settings) -> dict:
    caps = list_capabilities(settings)
    try:
        library_cfg = settings.library()
        default_library = verify_default_library(
            settings.default_library_path,
            str(library_cfg.get("sha256", "")),
            int(library_cfg.get("molecule_count", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        default_library = {"status": "unavailable", "reason": str(exc)}
    status = "available"
    if default_library["status"] == "unavailable":
        status = "unavailable"
    elif any(c.availability != "available" for c in caps):
        status = "degraded"
    return {
        "status": status,
        "config": str(settings.config_path),
        "database": str(settings.database_path),
        "task_root": str(settings.task_root),
        "default_library": default_library,
        "capabilities": [c.model_dump(mode="json") for c in caps],
    }
