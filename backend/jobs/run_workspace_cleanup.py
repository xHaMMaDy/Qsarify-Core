"""Retention cleanup for archived Studies and scheduled account deletion.

The job is deliberately service-side and idempotent. Study archives become
permanently removable after 30 days; an account deletion request purges
application-owned data only after the fixed 180-day retention window. Auth
identity deletion remains an operator/Supabase responsibility.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.target_intelligence_jobs import SupabaseWorkerStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expired_studies(store: SupabaseWorkerStore) -> list[dict[str, Any]]:
    rows = store.request("GET", "ti_workspaces", params={"select": "id,user_id,status,archive_expires_at", "status": "eq.archived", "archive_expires_at": f"lte.{_now()}"}) or []
    return [row for row in rows if row.get("status") == "archived" and row.get("id") and row.get("user_id")]


def expired_account_deletions(store: SupabaseWorkerStore) -> list[dict[str, Any]]:
    rows = store.request("GET", "ti_account_deletion_requests", params={"select": "id,user_id,status,purge_after,attempt_count", "status": "eq.scheduled", "purge_after": f"lte.{_now()}"}) or []
    return [row for row in rows if row.get("id") and row.get("user_id") and row.get("status") == "scheduled"]


def _safe_path(candidate: str | Path, root: Path) -> Path | None:
    try:
        resolved = Path(candidate).resolve()
        resolved.relative_to(root)
        return resolved
    except (OSError, ValueError):
        return None


def _uploaded_model_path(root: Path, user_id: str, model_id: str, artifact_format: str) -> Path | None:
    try:
        user_uuid = uuid.UUID(str(user_id))
        model_uuid = uuid.UUID(str(model_id))
    except (ValueError, AttributeError):
        return None
    extension = ".joblib" if str(artifact_format).lower() == "joblib" else ".pkl"
    return _safe_path(root / f"user_{user_uuid}" / "uploaded-models" / f"model_{model_uuid}" / f"model{extension}", root)


def _remove_files(paths: list[Path], root: Path) -> None:
    for candidate in paths:
        safe = _safe_path(candidate, root)
        if safe and safe.is_file():
            try:
                safe.unlink()
            except OSError:
                pass


def purge_account_data(store: SupabaseWorkerStore, user_id: str, root: Path) -> None:
    """Delete application records after retention, then remove owned files."""
    artifact_rows = store.request("GET", "ti_model_artifacts", params={"select": "artifact_path", "user_id": f"eq.{user_id}"}) or []
    uploaded_rows = store.request("GET", "ti_uploaded_models", params={"select": "id,artifact_format", "user_id": f"eq.{user_id}"}) or []

    # Delete direct owner records first. Workspace/report child records either
    # are included explicitly or are removed by their existing CASCADE FKs.
    owner_tables = (
        "ti_model_deployments", "ti_model_artifacts", "ti_training_runs",
        "ti_training_datasets", "ti_collection_jobs", "ti_study_plan_versions",
        "ti_study_plans", "ti_workspace_targets", "ti_workspace_members",
        "ti_handoffs", "ti_messages", "ti_uploads", "ti_reports", "ti_runs",
        "ti_usage_events", "ti_uploaded_models",
    )
    for table in owner_tables:
        store.request("DELETE", table, params={"user_id": f"eq.{user_id}"})
    store.request("DELETE", "ti_audit_events", params={"actor_user_id": f"eq.{user_id}"})
    store.request("DELETE", "ti_workspaces", params={"user_id": f"eq.{user_id}"})

    files: list[Path] = []
    files.extend(Path(str(row.get("artifact_path"))) for row in artifact_rows if row.get("artifact_path"))
    files.extend(path for row in uploaded_rows if (path := _uploaded_model_path(root, user_id, str(row.get("id")), str(row.get("artifact_format") or "pkl"))))
    _remove_files(files, root)
    for path in files:
        _remove_files([Path(str(path) + ".manifest.json")], root)


def cleanup_once(*, dry_run: bool = True, store: SupabaseWorkerStore | None = None) -> list[str]:
    store = store or SupabaseWorkerStore()
    if not dry_run:
        store.request("DELETE", "ti_public_prediction_buckets", params={"window_started_at": f"lt.{datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()}"})
    root = Path(os.environ.get("QSARIFY_MODEL_ARTIFACT_ROOT", Path(__file__).resolve().parents[1] / "models")).resolve()
    removed: list[str] = []

    for study in expired_studies(store):
        study_id = str(study["id"])
        if dry_run:
            removed.append("study:" + study_id)
            continue
        artifacts = store.request("GET", "ti_model_artifacts", params={"select": "artifact_path", "workspace_id": f"eq.{study_id}", "user_id": f"eq.{study['user_id']}"}) or []
        deleted = store.request("DELETE", "ti_workspaces", params={"id": f"eq.{study_id}", "status": "eq.archived", "archive_expires_at": f"lte.{_now()}"}) or []
        if deleted:
            files = [Path(str(item.get("artifact_path"))) for item in artifacts if item.get("artifact_path")]
            _remove_files(files, root)
            for path in files:
                _remove_files([Path(str(path) + ".manifest.json")], root)
            removed.append("study:" + study_id)

    for request in expired_account_deletions(store):
        request_id = str(request["id"])
        if dry_run:
            removed.append("account:" + request_id)
            continue
        try:
            purge_account_data(store, str(request["user_id"]), root)
            store.request("PATCH", "ti_account_deletion_requests", params={"id": f"eq.{request_id}", "status": "eq.scheduled"}, payload={"status": "executed", "executed_at": _now(), "last_error": None})
            removed.append("account:" + request_id)
        except Exception as exc:
            store.request("PATCH", "ti_account_deletion_requests", params={"id": f"eq.{request_id}"}, payload={"status": "failed", "attempt_count": int(request.get("attempt_count") or 0) + 1, "last_error": str(exc)[:500]})

    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="apply only expired retention actions")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=86400)
    args = parser.parse_args()
    while True:
        removed = cleanup_once(dry_run=not args.apply)
        print({"dry_run": not args.apply, "expired_items": removed}, flush=True)
        if not args.loop:
            return 0
        time.sleep(max(300, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
