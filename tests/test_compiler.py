import pytest

from autovs.compiler import choose_executable_strategy, compile_strategy
from autovs.schemas import ActionType, InputManifest, LibraryAsset, TargetAsset


def _manifest(*, target_locked=True):
    return InputManifest(
        query="a sufficiently detailed screening request",
        library_asset=LibraryAsset(source="user", path="library.smi", sha256="a" * 64),
        target_asset=TargetAsset(source="user" if target_locked else "research", locked=target_locked,
                                 path="protein.pdb" if target_locked else None, sha256="b" * 64 if target_locked else None),
    )


def test_compiler_rejects_unregistered_science_action():
    with pytest.raises(ValueError, match="unsupported"):
        compile_strategy({"strategy_name": "bad", "pipeline": [{"step_id": "x", "action_type": "covalent_docking"}]})


def test_compiler_adds_reproducibility_steps():
    plan = compile_strategy({"strategy_name": "ok", "pipeline": [{"step_id": "dock", "action_type": "molecular_docking"}]})
    actions = [step.action_type for step in plan.steps]
    assert ActionType.INPUT_VALIDATION in actions
    assert ActionType.FINAL_RANKING in actions
    assert ActionType.REPORT_GENERATION in actions


def test_compiler_discards_llm_authored_pocket_step():
    plan = compile_strategy({"strategy_name": "safe", "pipeline": [
        {"step_id": "invented-pocket", "action_type": "binding_site_detection",
         "parameters": {"center": [999, 999, 999]}},
        {"step_id": "dock", "action_type": "molecular_docking"},
    ]})
    pocket_steps = [step for step in plan.steps if step.action_type == ActionType.POCKET_DEFINITION]
    assert len(pocket_steps) == 1
    assert pocket_steps[0].step_id == "pocket-definition"
    assert pocket_steps[0].parameters == {}


def test_compiler_binds_symbolic_assets_and_rejects_external_library():
    plan = compile_strategy({"strategy_name": "safe", "pipeline": [
        {"step_id": "dock", "action_type": "molecular_docking"},
    ]}, input_manifest=_manifest())
    docking = next(step for step in plan.steps if step.action_type == ActionType.MOLECULAR_DOCKING)
    assert {item.name for item in docking.inputs} == {"prepared_library", "target_structure"}
    with pytest.raises(ValueError, match="external library"):
        compile_strategy({"strategy_name": "bad", "pipeline": [
            {"step_id": "filter", "action_type": "physicochemical_filtering",
             "description": "replace the input with ZINC"},
        ]}, input_manifest=_manifest())


def test_compiler_only_adds_docking_prerequisites_when_needed():
    plan = compile_strategy({"strategy_name": "prep-only", "pipeline": [
        {"step_id": "chembl", "action_type": "molecule_standardization_v2"},
        {"step_id": "ion", "action_type": "ionization_enumeration"},
    ]}, input_manifest=_manifest())
    actions = [step.action_type for step in plan.steps]
    assert ActionType.PROTEIN_PREPARATION not in actions
    assert ActionType.POCKET_DEFINITION not in actions
    assert ActionType.MOLECULE_STANDARDIZATION not in actions


def test_compiler_accepts_new_molecule_tool_chain_for_docking():
    plan = compile_strategy({"strategy_name": "new-tools", "pipeline": [
        {"step_id": "chembl", "action_type": "molecule_standardization_v2"},
        {"step_id": "gypsum", "action_type": "ligand_3d_enumeration"},
        {"step_id": "dock", "action_type": "molecular_docking"},
    ]}, input_manifest=_manifest())
    actions = [step.action_type for step in plan.steps]
    assert ActionType.MOLECULE_STANDARDIZATION not in actions
    assert ActionType.PROTEIN_PREPARATION in actions
    assert ActionType.POCKET_DEFINITION in actions
    assert actions.index(ActionType.LIGAND_3D_ENUMERATION) < actions.index(ActionType.MOLECULAR_DOCKING)


def test_compiler_rejects_service_owned_acquisition_and_absolute_paths():
    with pytest.raises(ValueError, match="service-owned"):
        compile_strategy({"strategy_name": "bad", "pipeline": [
            {"step_id": "download", "action_type": "target_structure_acquisition"},
        ]}, input_manifest=_manifest(target_locked=False))
    with pytest.raises(ValueError, match="absolute"):
        compile_strategy({"strategy_name": "bad", "pipeline": [
            {"step_id": "dock", "action_type": "molecular_docking", "parameters": {"library": "/tmp/other.smi"}},
        ]}, input_manifest=_manifest())


def test_ranked_strategy_falls_back_after_binding_violation():
    strategies = [
        {"strategy_name": "top", "pipeline": [{"step_id": "f", "action_type": "physicochemical_filtering", "description": "use Enamine"}]},
        {"strategy_name": "fallback", "pipeline": [{"step_id": "d", "action_type": "molecular_docking"}]},
    ]
    selected, plan, rejected = choose_executable_strategy(
        ["top", "fallback"], strategies, input_manifest=_manifest(),
    )
    assert selected["strategy_name"] == "fallback"
    assert plan.strategy_id == "fallback"
    assert rejected[0]["strategy_name"] == "top"
