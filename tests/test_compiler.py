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

