"""Durable Target Modeling Workspace training jobs.

The web endpoint only claims/starts a job. The actual worker persists every
stage in Supabase so a retry or a separate worker process can recover without
trusting browser state.
"""

from __future__ import annotations

import threading
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .target_intelligence_jobs import SupabaseWorkerStore
from .workspace_training import MODEL_NAMES, train_workspace_dataset


class TrainingCancelled(Exception):
    """Raised when the durable run receives a cancellation request."""


_active_runs: set[str] = set()
_active_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(store: SupabaseWorkerStore, run_id: str) -> dict[str, Any] | None:
    rows = store.request("GET", "ti_training_runs", params={"id": f"eq.{run_id}", "limit": "1"}) or []
    return rows[0] if rows else None


def _update(store: SupabaseWorkerStore, run_id: str, payload: Mapping[str, Any]) -> None:
    store.request("PATCH", "ti_training_runs", params={"id": f"eq.{run_id}"}, payload=dict(payload))


def _set_child_status(store: SupabaseWorkerStore, child_id: str, status: str, **extra: Any) -> None:
    _update(store, child_id, {"status": status, "progress_stage": status, **extra})


def _check_cancel(store: SupabaseWorkerStore, run_id: str) -> None:
    run = _run(store, run_id)
    if not run:
        raise ValueError("Training run not found")
    if bool(run.get("cancel_requested")) or run.get("status") in {"cancelling", "cancelled"}:
        _update(store, run_id, {"status": "cancelled", "progress_stage": "cancelled", "completed_at": _now(), "last_error": "Cancelled by user"})
        raise TrainingCancelled()
    expires_at = run.get("expires_at")
    if expires_at:
        try:
            if datetime.fromisoformat(str(expires_at).replace("Z", "+00:00")) < datetime.now(timezone.utc):
                _update(store, run_id, {"status": "expired", "progress_stage": "expired", "completed_at": _now(), "last_error": "Training run expired"})
                raise TrainingCancelled()
        except ValueError:
            pass


def _target_identity(target: Mapping[str, Any]) -> str:
    snapshot = target.get("target_snapshot") if isinstance(target.get("target_snapshot"), Mapping) else {}
    identifiers = snapshot.get("identifiers") if isinstance(snapshot.get("identifiers"), Mapping) else {}
    return str(identifiers.get("uniprot") or target.get("target_key") or target.get("id") or "")


def _load_training_inputs(store: SupabaseWorkerStore, run: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_ids = [str(value) for value in (run.get("dataset_ids") or []) if isinstance(value, str)]
    if not dataset_ids:
        raise ValueError("Training run contains no dataset IDs")
    datasets = store.get_training_datasets_for_run(dataset_ids, str(run["workspace_id"]), str(run["user_id"]))
    if len(datasets) != len(set(dataset_ids)):
        raise ValueError("One or more training datasets are unavailable")
    target_ids = sorted({str(row.get("workspace_target_id")) for row in datasets if row.get("workspace_target_id")})
    targets = store.request("GET", "ti_workspace_targets", params={
        "select": "id,target_key,target_snapshot",
        "id": f"in.({','.join(target_ids)})",
        "workspace_id": f"eq.{run['workspace_id']}",
        "user_id": f"eq.{run['user_id']}",
    }) or []
    target_by_id = {str(row["id"]): row for row in targets}
    prepared: list[dict[str, Any]] = []
    relevant_keys = ("curated_smiles", "smiles", "label", "bioactivity_class", "pIC50", "scaffold")
    for dataset in datasets:
        if dataset.get("status") != "ready_for_training":
            raise ValueError(f"Dataset {dataset.get('id')} is not ready_for_training")
        target = target_by_id.get(str(dataset.get("workspace_target_id")))
        if not target:
            raise ValueError("Training dataset target ownership could not be verified")
        records = dataset.get("curated_records") if isinstance(dataset.get("curated_records"), list) else dataset.get("records") or []
        identity = _target_identity(target)
        for record in records:
            if isinstance(record, Mapping):
                item = {key: record.get(key) for key in relevant_keys if key in record}
                item.setdefault("target_identity", identity)
                item.setdefault("workspace_target_id", target["id"])
                item.setdefault("target_key", target.get("target_key"))
                prepared.append(item)
    if not prepared:
        raise ValueError("Selected datasets contain no training records")
    return prepared


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    candidates = {}
    for name, candidate in (result.get("candidates") or {}).items():
        if not isinstance(candidate, Mapping):
            continue
        candidates[str(name)] = {"metrics": candidate.get("metrics") or {}, "ad": candidate.get("ad") or {}, "selection_score": candidate.get("selection_score")}
    return {
        "task_type": result.get("task_type"),
        "architecture": result.get("architecture"),
        "target_vocabulary": result.get("target_vocabulary") or [],
        "train_records": result.get("train_records"),
        "test_records": result.get("test_records"),
        "scaffold_count": result.get("scaffold_count"),
        "best_model": result.get("best_model"),
        "candidates": candidates,
        "feature_schema": result.get("feature_schema") or {},
    }


def _claim(store: SupabaseWorkerStore, run_id: str) -> dict[str, Any] | None:
    rows = store.request("PATCH", "ti_training_runs", params={"id": f"eq.{run_id}", "status": "eq.queued", "cancel_requested": "eq.false"}, payload={"status": "validating", "progress_stage": "validating", "progress_percent": 5, "started_at": _now()}) or []
    return rows[0] if rows else None


def process_training_run(run_id: str, store: SupabaseWorkerStore | None = None) -> dict[str, Any]:
    store = store or SupabaseWorkerStore()
    current = _run(store, run_id)
    if not current:
        raise ValueError("Training run not found")
    if current.get("status") == "completed":
        return current
    claimed = _claim(store, run_id)
    if not claimed:
        return _run(store, run_id) or {"id": run_id, "status": "skipped"}
    try:
        _check_cancel(store, run_id)
        architecture = str(claimed.get("architecture") or "")
        task_type = str(claimed.get("task_type") or "")
        model_names = [str(value) for value in (claimed.get("model_names") or []) if isinstance(value, str)]
        if task_type not in {"classification", "regression"} or architecture not in {"separate_models", "pooled_multitarget", "both"} or not model_names or any(value not in MODEL_NAMES for value in model_names):
            raise ValueError("Training run settings failed deterministic validation")
        records = _load_training_inputs(store, claimed)
        identities = sorted({str(row.get("target_identity") or "") for row in records if row.get("target_identity")})
        if architecture in {"pooled_multitarget", "both"} and len(identities) < 2:
            raise ValueError("Pooled training requires at least two selected target identities")
        _update(store, run_id, {"status": "training", "progress_stage": "training", "progress_percent": 20})
        _check_cancel(store, run_id)
        result_groups: list[tuple[str | None, list[Mapping[str, Any]], dict[str, Any], str]] = []
        scopes: list[tuple[str, str]] = [(architecture, run_id)]
        if architecture == "both":
            children = store.request("GET", "ti_training_runs", params={"parent_run_id": f"eq.{run_id}", "order": "created_at.asc"}) or []
            child_by_scope = {str(row.get("execution_scope")): str(row.get("id")) for row in children if row.get("id")}
            scopes = [("separate_models", child_by_scope.get("separate_child", run_id)), ("pooled_multitarget", child_by_scope.get("pooled_child", run_id))]
            for child_id in child_by_scope.values():
                _set_child_status(store, child_id, "training", progress_percent=20, started_at=_now())
        for scope, artifact_run_id in scopes:
            if scope == "pooled_multitarget":
                result = train_workspace_dataset(records, task_type=task_type, architecture=scope, model_names=model_names, workspace_id=str(claimed["workspace_id"]), owner_id=str(claimed["user_id"]), artifact_root=os.environ.get("QSARIFY_MODEL_ARTIFACT_ROOT", str(Path(__file__).resolve().parents[1] / "models")))
                result_groups.append((None, records, result, artifact_run_id))
                continue
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for record in records:
                grouped.setdefault(str(record.get("workspace_target_id") or record.get("target_identity") or ""), []).append(record)
            for index, (target_id, target_records) in enumerate(grouped.items(), start=1):
                _check_cancel(store, run_id)
                result = train_workspace_dataset(target_records, task_type=task_type, architecture=scope, model_names=model_names, workspace_id=str(claimed["workspace_id"]), owner_id=str(claimed["user_id"]), artifact_root=os.environ.get("QSARIFY_MODEL_ARTIFACT_ROOT", str(Path(__file__).resolve().parents[1] / "models")))
                result_groups.append((target_id, target_records, result, artifact_run_id))
                _update(store, run_id, {"progress_stage": "training", "progress_percent": min(70, 20 + int(index / max(len(grouped), 1) * 50))})
        _update(store, run_id, {"status": "evaluating", "progress_stage": "evaluating", "progress_percent": 80})
        _check_cancel(store, run_id)
        artifact_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for target_id, _target_records, result, artifact_run_id in result_groups:
            safe = _safe_result(result)
            summaries.append({"workspace_target_id": target_id, **safe})
            for model_family, candidate in (result.get("candidates") or {}).items():
                artifact_rows.append({
                    "workspace_id": claimed["workspace_id"],
                    "dataset_id": (claimed.get("dataset_ids") or [None])[0],
                    "training_dataset_ids": claimed.get("dataset_ids") or [],
                    "workspace_target_id": target_id,
                    "training_run_id": artifact_run_id,
                    "user_id": claimed["user_id"],
                    "task_type": result["task_type"],
                    "architecture": result["architecture"],
                    "model_family": model_family,
                    "status": "ready_for_review",
                    "metrics": candidate.get("metrics") or {},
                    "applicability_domain": candidate.get("ad") or {},
                    "model_card": {"task_type": result["task_type"], "architecture": result["architecture"], "feature_schema": result.get("feature_schema") or {}, "train_records": result.get("train_records"), "test_records": result.get("test_records"), "scaffold_count": result.get("scaffold_count"), "best_model": result.get("best_model"), "training_run_id": run_id, "training_dataset_ids": claimed.get("dataset_ids") or []},
                    "target_vocabulary": result.get("target_vocabulary") or [],
                    "artifact_path": candidate.get("artifact_path") or "",
                    "checksum_sha256": candidate.get("artifact_sha256"),
                    "artifact_size_bytes": candidate.get("artifact_size_bytes"),
                    "manifest": candidate.get("manifest") or {},
                    "storage_key": f"users/{claimed['user_id']}/studies/{claimed['workspace_id']}/models/{Path(str(candidate.get('artifact_path') or '')).name}",
                    "training_config_hash": claimed.get("confirmed_config_hash"),
                })
        saved = store.request("POST", "ti_model_artifacts", payload=artifact_rows) or []
        artifact_ids = [row.get("id") for row in saved if row.get("id")]
        summary = {"architecture": architecture, "task_type": task_type, "target_identities": identities, "groups": summaries}
        if architecture == "both":
            children = store.request("GET", "ti_training_runs", params={"parent_run_id": f"eq.{run_id}"}) or []
            for child in children:
                _set_child_status(store, str(child["id"]), "completed", progress_percent=100, completed_at=_now(), last_error=None)
        _update(store, run_id, {"status": "ready_for_review", "progress_stage": "ready_for_review", "progress_percent": 100, "completed_at": _now(), "result_summary": summary, "artifact_ids": artifact_ids, "last_error": None})
        return _run(store, run_id) or {"id": run_id, "status": "completed", "artifact_ids": artifact_ids}
    except TrainingCancelled:
        return _run(store, run_id) or {"id": run_id, "status": "cancelled"}
    except Exception as exc:
        _update(store, run_id, {"status": "failed", "progress_stage": "failed", "progress_percent": 100, "completed_at": _now(), "last_error": str(exc)[:500], "warnings": [{"code": "training_failure", "message": str(exc)[:500]}]})
        raise


def start_training_run(run_id: str) -> dict[str, Any]:
    with _active_lock:
        if run_id in _active_runs:
            return {"id": run_id, "status": "training", "already_running": True}
        _active_runs.add(run_id)

    def runner() -> None:
        try:
            process_training_run(run_id)
        except Exception:
            pass
        finally:
            with _active_lock:
                _active_runs.discard(run_id)

    threading.Thread(target=runner, name=f"qsarify-training-{run_id[:8]}", daemon=True).start()
    return {"id": run_id, "status": "queued", "started": True}


def process_pending_training_runs(limit: int = 5, store: SupabaseWorkerStore | None = None) -> list[dict[str, Any]]:
    store = store or SupabaseWorkerStore()
    rows = store.request("GET", "ti_training_runs", params={"status": "eq.queued", "order": "queued_at.asc", "limit": str(min(max(limit, 1), 20))}) or []
    results = []
    for row in rows:
        try:
            results.append(process_training_run(str(row["id"]), store=store))
        except Exception as exc:
            results.append({"id": row.get("id"), "status": "failed", "error": str(exc)})
    return results
