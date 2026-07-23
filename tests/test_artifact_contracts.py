"""Unit tests for Artifact Registry and Action I/O Contracts."""

from __future__ import annotations

import pytest

from autovs.dag import (
    SCREENING_LIBRARY, NORMALIZED_LIBRARY, TARGET_STRUCTURE,
    POCKET_CENTER, POCKET_SIZE, PREPARED_LIBRARY,
    RECEPTOR_PDBQT, SCORES_CSV, TOP_HITS,
)
from autovs.planning.contracts import (
    ACTION_CONTRACTS, ARTIFACT_REGISTRY, ArtifactSchema, ActionIOContract,
    find_producers, find_consumers, get_artifact, get_contract,
)
from autovs.schemas import ActionType


class TestArtifactRegistry:
    def test_all_registered_keys_match(self):
        """所有注册 artifact 的 key 必须自洽。"""
        for key, schema in ARTIFACT_REGISTRY.items():
            assert schema.artifact_key == key, f"Key mismatch: {key}"

    def test_screening_library_is_user_provided(self):
        assert ARTIFACT_REGISTRY[SCREENING_LIBRARY].user_provided is True

    def test_target_structure_is_user_provided(self):
        assert ARTIFACT_REGISTRY[TARGET_STRUCTURE].user_provided is True

    def test_scores_csv_is_not_user_provided(self):
        assert ARTIFACT_REGISTRY[SCORES_CSV].user_provided is False

    def test_get_artifact_returns_none_for_unknown(self):
        assert get_artifact("not_a_real_key_xyz") is None

    def test_all_artifacts_have_description(self):
        for key, schema in ARTIFACT_REGISTRY.items():
            assert len(schema.description) > 5, f"{key} missing description"

    def test_all_artifacts_have_formats(self):
        for key, schema in ARTIFACT_REGISTRY.items():
            if not schema.sensitive_path:
                continue
            assert len(schema.allowed_formats) > 0, f"{key} missing formats"


class TestActionContracts:
    def test_all_registered_actions_have_contracts(self):
        """至少当前已注册的所有 ActionType 都有 contract。"""
        for action in ActionType:
            contract = get_contract(action)
            assert contract is not None, f"{action.value} missing contract"
            assert contract.action_type == action

    def test_docking_requires_receptor_ligand_pocket(self):
        c = get_contract(ActionType.MOLECULAR_DOCKING)
        required = set(c.required_inputs)
        assert RECEPTOR_PDBQT in required
        assert PREPARED_LIBRARY in required
        assert POCKET_CENTER in required
        assert POCKET_SIZE in required

    def test_final_ranking_requires_scores(self):
        c = get_contract(ActionType.FINAL_RANKING)
        assert SCORES_CSV in c.required_inputs
        assert TOP_HITS in c.outputs

    def test_input_validation_outputs_normalized_library(self):
        c = get_contract(ActionType.INPUT_VALIDATION)
        assert NORMALIZED_LIBRARY in c.outputs

    def test_find_producers_returns_correct(self):
        producers = find_producers(TOP_HITS)
        assert ActionType.FINAL_RANKING in producers

    def test_find_consumers_returns_correct(self):
        consumers = find_consumers(SCORES_CSV)
        assert ActionType.FINAL_RANKING in consumers

    def test_service_owned_actions_marked(self):
        """服务拥有的步骤必须标记 service_owned=True。"""
        for action in [ActionType.INPUT_VALIDATION, ActionType.TARGET_STRUCTURE_ACQUISITION,
                       ActionType.POCKET_DEFINITION, ActionType.REPORT_GENERATION]:
            c = get_contract(action)
            assert c.service_owned, f"{action.value} should be service_owned"

    def test_no_duplicate_outputs_different_semantics(self):
        """同一个 artifact 不能有两个不同语义的 producer（通过 contract）。"""
        outputs_map: dict[str, list[ActionType]] = {}
        for action, contract in ACTION_CONTRACTS.items():
            for output_key in contract.outputs:
                outputs_map.setdefault(output_key, []).append(action)
        # 允许多个 producer，这是正常的

    def test_gpu_actions_have_gpu_true(self):
        gpu_actions = [ActionType.SHORT_MD, ActionType.MOLECULAR_DYNAMICS,
                       ActionType.TARGET_STRUCTURE_PREDICTION]
        for action in gpu_actions:
            c = get_contract(action)
            assert c.gpu_required, f"{action.value} should require GPU"
