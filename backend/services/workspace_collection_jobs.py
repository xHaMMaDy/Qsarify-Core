"""Durable per-target ChEMBL collection jobs for Study Workspace V2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from services.target_intelligence_jobs import SupabaseWorkerStore

_UNIPROT_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{3,19}$")
_ACTIVITY_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9 _-]{0,29}$")
_ACTIVE_STATUSES = {"queued", "validating", "collecting", "retrying"}


class CollectionCancelled(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _content_hash(records: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(_json_safe(records), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _update(store: SupabaseWorkerStore, job_id: str, payload: Mapping[str, Any]) -> None:
    store.request("PATCH", "ti_collection_jobs", params={"id": f"eq.{job_id}"}, payload=dict(payload))


def _event(store: SupabaseWorkerStore, job: Mapping[str, Any], event_type: str, *, status: str | None = None, percent: int | None = None, message: str | None = None, payload: Mapping[str, Any] | None = None) -> None:
    # Use the service role only to write a bounded public event. Raw records,
    # provider content, paths, and exceptions never enter this payload.
    store.request("POST", "ti_job_events", payload={
        "workspace_id": job["workspace_id"],
        "actor_user_id": job.get("user_id"),
        "job_id": job["id"],
        "entity_type": "collection_job",
        "event_type": event_type,
        "status": status,
        "progress_percent": percent,
        "safe_message": (message or "")[:500] or None,
        "public_payload": dict(payload or {}),
    })


def _check_cancel(store: SupabaseWorkerStore, job_id: str) -> None:
    rows = store.request("GET", "ti_collection_jobs", params={"id": f"eq.{job_id}", "select": "status,cancel_requested", "limit": "1"}) or []
    if not rows:
        raise ValueError("Collection job no longer exists")
    if rows[0].get("cancel_requested") is True or rows[0].get("status") in {"cancelling", "cancelled"}:
        raise CollectionCancelled("Collection was cancelled at a safe stage boundary")


def _claim(store: SupabaseWorkerStore, job_id: str) -> dict[str, Any] | None:
    rows = store.request(
        "PATCH",
        "ti_collection_jobs",
        params={"id": f"eq.{job_id}", "status": "in.(queued,retrying)", "cancel_requested": "eq.false"},
        payload={"status": "validating", "progress_stage": "validating", "progress_percent": 5, "started_at": _now()},
    ) or []
    return rows[0] if rows else None


def process_collection_job(job_id: str, store: SupabaseWorkerStore | None = None) -> dict[str, Any]:
    store = store or SupabaseWorkerStore()
    claimed = _claim(store, job_id)
    if not claimed:
        rows = store.request("GET", "ti_collection_jobs", params={"id": f"eq.{job_id}", "limit": "1"}) or []
        return rows[0] if rows else {"id": job_id, "status": "skipped"}

    try:
        _event(store, claimed, "status_changed", status="validating", percent=5, message="Validating target and collection settings")
        target_rows = store.request("GET", "ti_workspace_targets", params={
            "id": f"eq.{claimed['workspace_target_id']}",
            "workspace_id": f"eq.{claimed['workspace_id']}",
            "user_id": f"eq.{claimed['user_id']}",
            "select": "id,workspace_id,user_id,target_snapshot,report_id",
            "limit": "1",
        }) or []
        if not target_rows:
            raise ValueError("Study target is unavailable or is not owned by this Study")
        target = target_rows[0]
        snapshot = target.get("target_snapshot") or {}
        identifiers = snapshot.get("identifiers") if isinstance(snapshot, dict) else {}
        identifiers = identifiers if isinstance(identifiers, dict) else {}
        uniprot = str(identifiers.get("uniprot") or "").strip().upper()
        config = claimed.get("requested_config") if isinstance(claimed.get("requested_config"), dict) else {}
        activity_type = str(config.get("activity_type") or "IC50").strip().upper()
        threshold = float(config.get("threshold_nm") or 10000)
        if not _UNIPROT_PATTERN.fullmatch(uniprot):
            raise ValueError("A resolved human UniProt accession is required for ChEMBL collection")
        if not _ACTIVITY_PATTERN.fullmatch(activity_type):
            raise ValueError("Activity type is invalid")
        if not math.isfinite(threshold) or threshold <= 0 or threshold > 1_000_000_000:
            raise ValueError("Activity threshold must be a finite positive nM value")

        # Imported lazily to keep the web app import graph acyclic. These are
        # the existing, bounded, retrying ChEMBL helpers used by DataCollection.
        from app import get_all_bioactivities, get_all_molecules_concurrently, get_target, process_and_merge_data

        _check_cancel(store, job_id)
        _update(store, job_id, {"status": "collecting", "progress_stage": "resolving_target", "progress_percent": 10})
        _event(store, claimed, "status_changed", status="collecting", percent=10, message="Resolving ChEMBL target")
        chembl_target = get_target(uniprot)
        target_chembl_id = str(chembl_target.get("target_chembl_id") or "").strip()
        if not target_chembl_id:
            raise ValueError("ChEMBL target mapping is unavailable for this UniProt accession")

        _check_cancel(store, job_id)
        _update(store, job_id, {"progress_stage": "collecting_activities", "progress_percent": 25})
        _event(store, claimed, "progress", status="collecting", percent=25, message="Collecting ChEMBL activity records")
        activities: list[dict[str, Any]] = []
        for progress in get_all_bioactivities(target_chembl_id, activity_type):
            if isinstance(progress, str):
                continue
            activities = progress
        if not activities:
            raise ValueError("No ChEMBL activity records were found for this target and endpoint")

        _check_cancel(store, job_id)
        _update(store, job_id, {"progress_stage": "collecting_molecules", "progress_percent": 55, "record_count": len(activities)})
        _event(store, claimed, "progress", status="collecting", percent=55, message=f"Resolving metadata for {len(activities)} activity records")
        molecule_ids = [activity.get("molecule_chembl_id") for activity in activities]
        molecules: dict[str, Any] = {}
        for progress in get_all_molecules_concurrently(molecule_ids):
            if isinstance(progress, str):
                continue
            molecules = progress

        _check_cancel(store, job_id)
        _update(store, job_id, {"progress_stage": "building_dataset", "progress_percent": 80})
        _event(store, claimed, "progress", status="collecting", percent=80, message="Building a versioned Study dataset")
        records = _json_safe(process_and_merge_data(activities, molecules, threshold))
        if not records:
            raise ValueError("ChEMBL records could not be converted into a dataset")
        digest = _content_hash(records)
        existing = store.request("GET", "ti_training_datasets", params={
            "workspace_target_id": f"eq.{claimed['workspace_target_id']}",
            "select": "dataset_version",
            "order": "dataset_version.desc",
            "limit": "1",
        }) or []
        version = int(existing[0].get("dataset_version") or 0) + 1 if existing else 1
        dataset_rows = store.request("POST", "ti_training_datasets", payload={
            "workspace_id": claimed["workspace_id"],
            "workspace_target_id": claimed["workspace_target_id"],
            "user_id": claimed["user_id"],
            "dataset_version": version,
            "status": "collected",
            "collection_config": {
                "uniprot_id": uniprot,
                "chembl_target_id": target_chembl_id,
                "activity_type": activity_type,
                "threshold_nm": threshold,
                "collected_at": _now(),
            },
            "record_count": len(records),
            "records": records,
            "content_sha256": digest,
            "source_snapshot": {"report_id": target.get("report_id"), "target_snapshot": snapshot, "chembl_target_id": target_chembl_id},
        })
        if not dataset_rows:
            raise RuntimeError("Unable to persist the collected Study dataset")
        dataset = dataset_rows[0]
        completed = {"status": "completed", "progress_stage": "completed", "progress_percent": 100, "record_count": len(records), "completed_at": _now(), "error_code": None, "error_message": None}
        _update(store, job_id, completed)
        _event(store, {**claimed, **completed}, "completed", status="completed", percent=100, message=f"Collected {len(records)} compounds", payload={"dataset_id": dataset.get("id"), "dataset_version": version, "record_count": len(records), "content_sha256": digest})
        return {**claimed, **completed, "dataset": {"id": dataset.get("id"), "dataset_version": version, "record_count": len(records), "content_sha256": digest}}
    except CollectionCancelled as exc:
        _update(store, job_id, {"status": "cancelled", "progress_stage": "cancelled", "progress_percent": 100, "completed_at": _now(), "error_code": "cancelled", "error_message": str(exc)[:500]})
        return {"id": job_id, "status": "cancelled", "error_code": "cancelled"}
    except Exception as exc:
        error_code = "source_unavailable" if isinstance(exc, (OSError, ConnectionError)) else "collection_failed"
        _update(store, job_id, {"status": "failed", "progress_stage": "failed", "progress_percent": 100, "completed_at": _now(), "error_code": error_code, "error_message": str(exc)[:500]})
        _event(store, claimed, "failed", status="failed", percent=100, message="Collection failed; review the error and retry if appropriate", payload={"code": error_code})
        return {"id": job_id, "status": "failed", "error_code": error_code, "error_message": str(exc)[:500]}


def start_collection_job(job_id: str) -> dict[str, Any]:
    def runner() -> None:
        process_collection_job(job_id)

    threading.Thread(target=runner, name=f"qsarify-collection-{job_id[:8]}", daemon=True).start()
    return {"id": job_id, "status": "collecting", "started": True}


def process_pending_collection_jobs(limit: int = 5, store: SupabaseWorkerStore | None = None) -> list[dict[str, Any]]:
    store = store or SupabaseWorkerStore()
    rows = store.request("GET", "ti_collection_jobs", params={
        "status": "in.(queued,retrying)",
        "cancel_requested": "eq.false",
        "order": "queued_at.asc",
        "limit": str(min(max(limit, 1), 20)),
    }) or []
    return [process_collection_job(str(row["id"]), store=store) for row in rows if row.get("id")]

