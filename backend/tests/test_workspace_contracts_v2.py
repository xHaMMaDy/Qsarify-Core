import json
from pathlib import Path

import pytest

from services import workspace_contracts as contracts


FIXTURE = Path(__file__).parent / "fixtures" / "workspace_v2_contracts.json"


def test_workspace_v2_contract_fixture_matches_authoritative_constants():
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert frozen["contract_version"] == contracts.CONTRACT_VERSION
    assert frozen["feature_flag"] == contracts.FEATURE_FLAG_KEY
    assert frozen["limits"] == {
        "max_targets_per_study": contracts.MAX_TARGETS_PER_STUDY,
        "low_activity_warning_threshold": contracts.LOW_ACTIVITY_WARNING_THRESHOLD,
    }
    assert frozen["study_steps"] == list(contracts.STUDY_STEPS)
    assert frozen["roles"] == list(contracts.WORKSPACE_ROLES)
    assert frozen["study_statuses"] == list(contracts.STUDY_STATUSES)
    assert frozen["collection_job_statuses"] == list(contracts.COLLECTION_JOB_STATUSES)
    assert frozen["dataset_statuses"] == list(contracts.DATASET_STATUSES)
    assert frozen["training_run_statuses"] == list(contracts.TRAINING_RUN_STATUSES)
    assert frozen["model_statuses"] == list(contracts.MODEL_STATUSES)
    assert frozen["deployment_statuses"] == list(contracts.DEPLOYMENT_STATUSES)


@pytest.mark.parametrize(
    ("role", "permission", "allowed"),
    [
        ("owner", "workspace.delete", True),
        ("admin", "members.manage", True),
        ("editor", "jobs.start", True),
        ("editor", "candidates.review", False),
        ("reviewer", "candidates.review", True),
        ("reviewer", "plans.edit", False),
        ("viewer", "workspace.read", True),
        ("viewer", "artifacts.download", False),
        ("unknown", "workspace.read", False),
    ],
)
def test_workspace_role_permissions_fail_closed(role, permission, allowed):
    assert contracts.role_can(role, permission) is allowed


def test_workspace_transition_contract_accepts_expected_path():
    path = [
        "draft",
        "planning",
        "ready_for_collection",
        "collecting",
        "data_review",
        "ready_for_training",
        "training",
        "candidate_review",
        "ready_for_deployment",
        "deploying",
        "ready",
    ]
    for current, target in zip(path, path[1:]):
        contracts.validate_study_transition(current, target)


def test_workspace_transition_contract_rejects_skipping_gates():
    with pytest.raises(ValueError, match="cannot transition"):
        contracts.validate_study_transition("draft", "deploying")


def test_deleted_study_is_terminal():
    with pytest.raises(ValueError, match="cannot transition"):
        contracts.validate_study_transition("deleted", "draft")

