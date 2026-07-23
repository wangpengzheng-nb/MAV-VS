"""Planning-specific exceptions for ToolUsePlannerAgent."""

from __future__ import annotations


class PlannerError(RuntimeError):
    """Base exception for all planning errors."""


class PlannerCapabilityGapError(PlannerError):
    """A required capability is unavailable and no alternative exists."""

    def __init__(self, action_type: str, reason: str):
        super().__init__(f"capability gap: {action_type} — {reason}")
        self.action_type = action_type
        self.reason = reason


class ArtifactGapError(PlannerError):
    """No producer exists for a required artifact."""

    def __init__(self, artifact_key: str, consumer_action: str):
        super().__init__(
            f"artifact gap: '{artifact_key}' required by {consumer_action} "
            f"has no available producer"
        )
        self.artifact_key = artifact_key
        self.consumer_action = consumer_action


class PlanningValidationError(PlannerError):
    """The generated plan fails validation checks."""


class AssetLockViolation(PlannerError):
    """Strategy attempts to replace a locked user asset."""

    def __init__(self, asset_name: str, detail: str = ""):
        super().__init__(f"asset lock violation: {asset_name}" + (f" — {detail}" if detail else ""))
        self.asset_name = asset_name
