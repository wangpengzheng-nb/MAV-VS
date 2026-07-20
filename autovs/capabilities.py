from __future__ import annotations

import shutil
from pathlib import Path

from autovs.config import Settings
from autovs.schemas import ActionType, ToolCapability


CAPABILITY_DEFINITIONS = {
    ActionType.INPUT_VALIDATION: ("Input validator", "Validate PDB and molecular library inputs", "python", ["PDB", "SMI", "CSV", "SDF"], ["JSON"], False),
    ActionType.PROTEIN_PREPARATION: ("OpenBabel protein preparation", "Remove water, add hydrogens and produce receptor files", "conda", ["PDB"], ["PDB", "PDBQT"], False),
    ActionType.POCKET_DEFINITION: ("Evidence-backed pocket resolver", "Resolve and validate a pocket from a user box, uploaded cocrystal ligand, verified research structure, or mapped key residues", "python", ["PDB", "JSON"], ["JSON"], False),
    ActionType.MOLECULE_STANDARDIZATION: ("RDKit standardization", "Canonicalize, deduplicate, filter and assign stable IDs", "python", ["SMI", "CSV", "SDF"], ["CSV", "SDF"], False),
    ActionType.CONFORMER_GENERATION: ("RDKit ETKDGv3", "Generate explicit-H 3D conformers with MMFF94s/UFF", "python", ["SMI", "CSV", "SDF"], ["SDF", "CSV"], False),
    ActionType.PHYSICOCHEMICAL_FILTERING: ("RDKit filters", "Apply physicochemical, PAINS and reactive-group gates", "python", ["SMI", "CSV", "SDF"], ["CSV", "SDF"], False),
    ActionType.DIVERSITY_SELECTION: ("Murcko diversity selector", "Limit scaffold monopolization while preserving rank", "python", ["CSV", "SDF"], ["CSV", "SDF"], False),
    ActionType.MOLECULAR_DOCKING: ("smina/GNINA docking", "CPU smina rough docking or GPU GNINA rough/refinement", "slurm", ["PDB", "PDBQT", "SDF", "JSON"], ["SDF", "CSV"], False),
    ActionType.POSE_EXTRACTION: ("Docking pose extractor", "Select best affinity or best CNN_VS pose per molecule", "python", ["SDF"], ["SDF", "CSV"], False),
    ActionType.INTERACTION_ANALYSIS: ("PLIP", "Compute protein-ligand interaction fingerprints", "conda", ["PDB"], ["XML", "TXT", "CSV"], False),
    ActionType.ADMET_FILTERING: ("ADMET-AI", "Predict ADMET risks and physicochemical properties", "conda", ["CSV"], ["CSV"], False),
    ActionType.SHORT_MD: ("GROMACS 10 ns quality gate", "Charge-audited short MD stability check", "apptainer", ["PDB", "SDF", "CSV"], ["XTC", "CSV", "JSON"], True),
    ActionType.MOLECULAR_DYNAMICS: ("GROMACS 100 ns + MMGBSA", "Charge-audited production MD and 70-100 ns MMGBSA", "apptainer", ["PDB", "SDF", "CSV"], ["XTC", "CSV", "JSON"], True),
    ActionType.FINAL_RANKING: ("Evidence ranker", "Direction-aware normalized ranking with scaffold diversity", "python", ["CSV", "SDF"], ["CSV", "SDF"], False),
    ActionType.REPORT_GENERATION: ("Reproducibility reporter", "Generate Markdown, HTML and artifact manifest", "python", ["JSON", "CSV"], ["MD", "HTML", "JSON"], False),
}


def _exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def list_capabilities(settings: Settings) -> list[ToolCapability]:
    result: list[ToolCapability] = []
    for action, definition in CAPABILITY_DEFINITIONS.items():
        name, desc, executor, inputs, outputs, gpu = definition
        availability, reason = "available", ""
        if action == ActionType.MOLECULAR_DOCKING:
            smina, gnina = settings.executable("smina"), settings.executable("gnina")
            if not _exists(smina) and not _exists(gnina):
                availability, reason = "unavailable", "neither smina nor GNINA is configured"
            elif not _exists(gnina):
                availability, reason = "degraded", "GNINA unavailable; CPU smina only"
        elif action == ActionType.POCKET_DEFINITION and not _exists(settings.executable("plip")):
            availability, reason = "degraded", "PLIP unavailable; geometric ligand-contact validation remains available"
        elif action == ActionType.INTERACTION_ANALYSIS and not _exists(settings.executable("plip")):
            availability, reason = "unavailable", "PLIP binary not found"
        elif action == ActionType.PROTEIN_PREPARATION and not _exists(settings.executable("obabel")):
            availability, reason = "unavailable", "OpenBabel binary not found"
        elif action == ActionType.ADMET_FILTERING:
            conda = settings.executable("conda")
            env_path = conda.parent.parent / "envs" / settings.environment("admet") if conda else None
            if not _exists(env_path):
                availability, reason = "degraded", "autovs-admet environment is not installed"
        elif action in {ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS}:
            if not _exists(settings.container("gromacs")):
                availability, reason = "unavailable", "GROMACS Apptainer image not found"
            elif not _exists(settings.executable("sbatch")):
                availability, reason = "unavailable", "Slurm sbatch not found"
            else:
                availability, reason = "degraded", "GPU execution requires a healthy Slurm GPU partition"
        result.append(ToolCapability(
            action_type=action, name=name, description=desc, availability=availability,
            executor=executor, input_formats=inputs, output_formats=outputs,
            gpu_required=gpu, reason=reason,
        ))
    return result


def health_report(settings: Settings) -> dict:
    caps = list_capabilities(settings)
    status = "available"
    if any(c.availability != "available" for c in caps):
        status = "degraded"
    return {
        "status": status,
        "config": str(settings.config_path),
        "database": str(settings.database_path),
        "task_root": str(settings.task_root),
        "capabilities": [c.model_dump(mode="json") for c in caps],
    }
