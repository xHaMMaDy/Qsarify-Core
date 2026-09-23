"""Durable Target Intelligence run processing through Supabase REST.

The worker uses real Supabase and public-source APIs. It is intentionally a
separate process entry point so web requests do not own long-running work.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Mapping

import requests
from dotenv import load_dotenv

from services.target_intelligence import SourceRecord, run_target_intelligence_search
from services.literature_uploads import extract_literature_text
from services.reference_sources import resolve_reference_inputs

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class SupabaseWorkerStore:
    def __init__(self, base_url: str | None = None, service_role_key: str | None = None):
        self.base_url = (base_url or os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
        self.service_role_key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not self.base_url or not self.service_role_key:
            raise RuntimeError("SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, *, params: Mapping[str, Any] | None = None, payload: Any = None) -> Any:
        response = requests.request(
            method,
            f"{self.base_url}/rest/v1/{path.lstrip('/')}",
            params=params,
            headers={**self.headers, "Prefer": "return=representation"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.request("GET", "ti_runs", params={"id": f"eq.{run_id}", "limit": 1})
        return rows[0] if rows else None

    def update_run(self, run_id: str, payload: Mapping[str, Any]) -> None:
        self.request("PATCH", "ti_runs", params={"id": f"eq.{run_id}"}, payload=dict(payload))

    def claim_run(self, run_id: str) -> dict[str, Any] | None:
        """Atomically move a queued run out of ``created`` state."""
        rows = self.request(
            "PATCH",
            "ti_runs",
            params={"id": f"eq.{run_id}", "status": "eq.created"},
            payload={"status": "validating", "progress_stage": "validating", "progress_percent": 5},
        ) or []
        return rows[0] if rows else None

    def insert_report(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        rows = self.request("POST", "ti_reports", payload=dict(payload))
        return rows[0] if rows else {}

    def get_uploads(self, upload_ids: list[str], user_id: str) -> list[dict[str, Any]]:
        if not upload_ids:
            return []
        return self.request(
            "GET",
            "ti_uploads",
            params={"id": f"in.({','.join(upload_ids)})", "user_id": f"eq.{user_id}"},
        ) or []

    def download_upload(self, storage_path: str) -> bytes:
        response = requests.get(
            f"{self.base_url}/storage/v1/object/target-intelligence/{storage_path}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.content


def load_upload_sources(store: SupabaseWorkerStore, upload_ids: list[str], user_id: str, warnings: list[dict[str, str]] | None = None) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    for row in store.get_uploads(upload_ids, user_id):
        try:
            text = extract_literature_text(store.download_upload(row["storage_path"]), row["content_type"])
            sources.append(SourceRecord(
                source_type="upload",
                stable_id=f"UPLOAD:{row['id']}",
                title=row.get("original_filename"),
                evidence_tier="D",
                abstract_or_excerpt=text,
                retrieved_at=row.get("created_at"),
                source_snapshot_hash=row.get("sha256"),
                metadata={"upload_id": row["id"], "content_type": row.get("content_type"), "owner_id": user_id},
            ))
        except Exception as exc:
            if warnings is not None:
                warnings.append({"source_type": "upload", "error": f"Unable to extract {row.get('original_filename', 'uploaded file')}: {str(exc)[:300]}"})
            continue
    return sources


def process_run(run_id: str, store: SupabaseWorkerStore | None = None) -> dict[str, Any]:
    store = store or SupabaseWorkerStore()
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"Target Intelligence run not found: {run_id}")
    if run.get("status") in {"completed", "cancelled", "expired"}:
        return run
    if run.get("status") == "created":
        claimed = store.claim_run(run_id)
        if not claimed:
            return {"run_id": run_id, "status": "skipped", "reason": "claimed_by_other_worker"}
        run = claimed

    try:
        upload_ids = [str(value) for value in (run.get("upload_ids") or []) if isinstance(value, str)]
        upload_errors: list[dict[str, str]] = []
        upload_sources = load_upload_sources(store, upload_ids, str(run["user_id"]), upload_errors)
        reference_inputs = [str(value) for value in (run.get("reference_inputs") or []) if isinstance(value, str)]
        reference_sources, reference_errors = resolve_reference_inputs(reference_inputs)
        attempt_count = int(run.get("attempt_count") or 0) + 1
        store.update_run(run_id, {"status": "retrieving", "progress_stage": "retrieving", "progress_percent": 15, "attempt_count": attempt_count, "last_error": None})
        report = run_target_intelligence_search(
            str(run["question"]),
            ranking_weights=run.get("ranking_weights"),
            source_types=run.get("requested_sources") or ("europe_pmc", "pubmed", "chembl", "uniprot", "openalex", "crossref"),
            max_targets=10,
            source_limit=10,
            additional_sources=[*upload_sources, *reference_sources],
        )
        report["provenance"]["source_errors"].extend(reference_errors)
        report["provenance"]["source_errors"].extend(upload_errors)
        if upload_errors:
            report["limitations"].append("One or more private literature files could not be extracted and were excluded from evidence context.")
        if any(source.metadata.get("input_kind") == "publisher_url" for source in reference_sources):
            report["limitations"].append("Publisher URLs are recorded as provenance but are not fetched automatically.")
        store.update_run(run_id, {"status": "generating_report", "progress_stage": "generating_report", "progress_percent": 85})
        stored = store.insert_report({
            "run_id": run_id,
            "user_id": run["user_id"],
            "report_version": report["report_version"],
            "report": report,
        })
        store.update_run(run_id, {"status": "completed", "progress_stage": "completed", "progress_percent": 100})
        return {"run_id": run_id, "status": "completed", "report_id": stored.get("id"), "report": report}
    except Exception as exc:
        store.update_run(run_id, {
            "status": "failed",
            "progress_stage": "failed",
            "last_error": str(exc)[:500],
            "warnings": [{"code": "worker_failure", "message": str(exc)[:500]}],
        })
        raise


def process_pending_runs(limit: int = 10, store: SupabaseWorkerStore | None = None) -> list[dict[str, Any]]:
    store = store or SupabaseWorkerStore()
    rows = store.request(
        "GET",
        "ti_runs",
        params={"status": "eq.created", "order": "created_at.asc", "limit": min(max(limit, 1), 50)},
    ) or []
    results = []
    for row in rows:
        expires_at = row.get("expires_at")
        if expires_at:
            try:
                expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                if expiry < datetime.now(timezone.utc):
                    store.update_run(row["id"], {"status": "expired", "progress_stage": "expired", "last_error": "Run expired before processing"})
                    results.append({"run_id": row.get("id"), "status": "expired"})
                    continue
            except ValueError:
                pass
        try:
            claimed = store.claim_run(row["id"])
            if not claimed:
                results.append({"run_id": row.get("id"), "status": "skipped", "reason": "claimed_by_other_worker"})
                continue
            results.append(process_run(row["id"], store=store))
        except Exception as exc:
            results.append({"run_id": row.get("id"), "status": "failed", "error": str(exc)})
    return results
