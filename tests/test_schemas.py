import pytest
from pydantic import ValidationError

from autovs.schemas import ActionType, WorkflowPlan, WorkflowStep


def test_workflow_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        WorkflowStep(step_id="x", action_type="input_validation", surprise=True)


def test_workflow_rejects_forward_dependency():
    with pytest.raises(ValidationError):
        WorkflowPlan(strategy_id="x", steps=[WorkflowStep(step_id="b", action_type=ActionType.INPUT_VALIDATION, requires=["a"])])


def test_workflow_accepts_ordered_dag():
    plan = WorkflowPlan(strategy_id="x", steps=[
        WorkflowStep(step_id="a", action_type=ActionType.INPUT_VALIDATION),
        WorkflowStep(step_id="b", action_type=ActionType.POCKET_DEFINITION, requires=["a"]),
    ])
    assert plan.plan_version == "1.0"

