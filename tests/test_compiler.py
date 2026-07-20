import pytest

from autovs.compiler import compile_strategy
from autovs.schemas import ActionType


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
