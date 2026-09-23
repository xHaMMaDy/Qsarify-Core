"""Authoritative Target Modeling Workspace V2 lifecycle and role contracts.

These constants are intentionally provider- and UI-independent. API routes,
workers, migrations, and tests share the same vocabulary so a client cannot
invent a state transition or permission by sending a different string.
"""

from __future__ import annotations

from collections.abc import Iterable


CONTRACT_VERSION = "workspace-v2-p0"
FEATURE_FLAG_KEY = "target_modeling_workspace_v2"
MAX_TARGETS_PER_STUDY = 20
LOW_ACTIVITY_WARNING_THRESHOLD = 50

STUDY_STEPS = (
    "overview",
    "plan",
    "data",
    "datasets",
    "training",
    "review",
    "deploy",
    "playground",
)

STUDY_STATUSES = (
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
    "archived",
    "deleting",
    "deleted",
    "failed",
    "cancelling",
    "cancelled",
)

# Existing rows can contain these values. They remain readable during the
# additive migration and are mapped to canonical V2 states by the V2 API.
LEGACY_STUDY_STATUSES = (
    "planned",
    "awaiting_confirmation",
    "curating",
    "evaluating",
    "awaiting_deployment_confirmation",
    "expired",
    "rejected",
    "not_ready",
)

WORKSPACE_ROLES = ("owner", "editor", "reviewer", "viewer", "admin")

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "owner": frozenset(
        {
            "workspace.read",
            "workspace.edit",
            "workspace.archive",
            "workspace.delete",
            "members.manage",
            "targets.manage",
            "plans.edit",
            "jobs.start",
            "jobs.cancel",
            "candidates.review",
            "deployments.manage",
            "artifacts.download",
        }
    ),
    "admin": frozenset(
        {
            "workspace.read",
            "workspace.edit",
            "workspace.archive",
            "members.manage",
            "targets.manage",
            "plans.edit",
            "jobs.start",
            "jobs.cancel",
            "candidates.review",
            "deployments.manage",
            "artifacts.download",
        }
    ),
    "editor": frozenset(
        {
            "workspace.read",
            "workspace.edit",
            "targets.manage",
            "plans.edit",
            "jobs.start",
            "jobs.cancel",
            "deployments.manage",
            "artifacts.download",
        }
    ),
    "reviewer": frozenset(
        {
            "workspace.read",
            "candidates.review",
            "artifacts.download",
        }
    ),
    "viewer": frozenset({"workspace.read"}),
}

STUDY_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"planning", "archived", "deleting", "failed"}),
    "planning": frozenset({"draft", "ready_for_collection", "archived", "failed"}),
    "ready_for_collection": frozenset({"planning", "collecting", "archived", "failed"}),
    "collecting": frozenset({"data_review", "cancelling", "failed"}),
    "data_review": frozenset({"collecting", "ready_for_training", "archived", "failed"}),
    "ready_for_training": frozenset({"data_review", "training", "archived", "failed"}),
    "training": frozenset({"candidate_review", "cancelling", "failed"}),
    "candidate_review": frozenset({"training", "ready_for_deployment", "archived", "failed"}),
    "ready_for_deployment": frozenset({"candidate_review", "deploying", "archived", "failed"}),
    "deploying": frozenset({"ready", "cancelling", "failed"}),
    "ready": frozenset({"planning", "archived", "failed"}),
    "failed": frozenset({"planning", "ready_for_collection", "ready_for_training", "archived", "deleting"}),
    "cancelling": frozenset({"cancelled", "failed"}),
    "cancelled": frozenset({"planning", "ready_for_collection", "ready_for_training", "archived", "deleting"}),
    "archived": frozenset({"draft", "planning", "data_review", "candidate_review", "ready", "deleting"}),
    "deleting": frozenset({"deleted"}),
    "deleted": frozenset(),
}

COLLECTION_JOB_STATUSES = (
    "queued",
    "validating",
    "collecting",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
    "retrying",
)

DATASET_STATUSES = (
    "planned",
    "collecting",
    "collected",
    "curating",
    "curated",
    "preparing",
    "ready_for_training",
    "failed",
    "superseded",
)

TRAINING_RUN_STATUSES = (
    "queued",
    "validating",
    "featurizing",
    "training",
    "evaluating",
    "ready_for_review",
    "failed",
    "retrying",
    "cancelling",
    "cancelled",
)

MODEL_STATUSES = (
    "training",
    "ready_for_review",
    "approved",
    "rejected",
    "deploying",
    "deployed",
    "disabled",
    "failed",
)

DEPLOYMENT_STATUSES = ("draft", "validating", "active", "disabled", "failed", "superseded")

JOB_EVENT_ENTITY_TYPES = (
    "study",
    "collection_job",
    "dataset",
    "training_run",
    "model_artifact",
    "deployment",
)


def role_can(role: str, permission: str) -> bool:
    """Return whether a validated Study role grants a permission."""

    return permission in ROLE_PERMISSIONS.get(str(role), frozenset())


def validate_study_transition(current: str, target: str) -> None:
    """Reject unknown or impossible canonical Study state transitions."""

    if current not in STUDY_TRANSITIONS:
        raise ValueError(f"Unknown current Study status: {current}")
    if target == current:
        return
    if target not in STUDY_TRANSITIONS[current]:
        raise ValueError(f"Study status cannot transition from {current} to {target}")


def validate_choice(value: str, allowed: Iterable[str], label: str) -> str:
    normalized = str(value or "").strip()
    if normalized not in set(allowed):
        raise ValueError(f"Unsupported {label}: {normalized or 'empty'}")
    return normalized


def validate_step(step: str) -> str:
    return validate_choice(step, STUDY_STEPS, "Study step")

