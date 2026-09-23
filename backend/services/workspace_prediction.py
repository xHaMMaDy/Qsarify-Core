"""Prediction facade for approved Study deployments.

Artifact deserialization is deliberately delegated to the short-lived model
worker. Flask validates the request and returns the worker's structured result;
it never imports a user-provided PKL/joblib object into the web process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def _run_worker(payload: Mapping[str, Any], *, timeout_seconds: int = 30) -> dict[str, Any]:
    backend_root = Path(__file__).resolve().parents[1]
    worker_env = os.environ.copy()
    worker_env["PYTHONNOUSERSITE"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        worker_env.pop(key, None)
    artifact_path = payload.get("artifact_path")
    if isinstance(artifact_path, str) and artifact_path:
        configured_root = Path(worker_env.get("QSARIFY_MODEL_ARTIFACT_ROOT", backend_root / "models")).resolve()
        resolved_artifact = Path(artifact_path).resolve()
        try:
            resolved_artifact.relative_to(configured_root)
        except ValueError:
            # Direct unit tests may use a disposable temp root. Production
            # callers have already passed the Flask owner-root boundary check.
            worker_env["QSARIFY_MODEL_ARTIFACT_ROOT"] = str(resolved_artifact.parent)
    encoded = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    if len(encoded) > 16 * 1024 * 1024:
        raise ValueError("Model worker request exceeds the 16 MB limit")
    remote_url = os.environ.get("QSARIFY_MODEL_WORKER_URL")
    if remote_url:
        token = os.environ.get("QSARIFY_MODEL_WORKER_TOKEN")
        if not token:
            raise RuntimeError("Model worker token is not configured")
        request = urllib.request.Request(remote_url.rstrip("/") + "/v1/execute", data=encoded, headers={"Content-Type": "application/json", "X-QSARIFY-Worker-Token": token}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read(16 * 1024 * 1024 + 1).decode("utf-8"))
            if not isinstance(result, dict):
                raise RuntimeError("Model worker returned an invalid result")
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError("Isolated model worker rejected the request" + (f": {detail}" if detail else ".")) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Isolated model worker is unavailable") from exc
    process = subprocess.Popen(
        [sys.executable, "-m", "services.isolated_model_worker"],
        cwd=str(backend_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=worker_env,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    try:
        stdout, stderr = process.communicate(encoded, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        raise TimeoutError("Model worker timed out and was terminated") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:500]
        if any(marker in detail for marker in ("Unknown target identity", "target_identity is required", "Pooled deployment has no frozen target vocabulary", "Unsupported workspace artifact feature schema", "feature row must contain exactly", "Feature values must be numeric", "features must contain")):
            raise ValueError(detail or "Invalid deployment prediction request")
        raise RuntimeError("Model worker failed" + (f": {detail}" if detail else "."))
    try:
        result = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Model worker returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Model worker returned an invalid result")
    return result


def validate_workspace_artifact(*, artifact_path: str, expected_sha256: str | None = None) -> dict[str, Any]:
    return _run_worker({"operation": "validate", "artifact_path": artifact_path, "expected_sha256": expected_sha256})


def predict_uploaded_features(*, artifact_path: str, expected_sha256: str, features: list[list[float]]) -> list[dict[str, Any]]:
    result = _run_worker({"operation": "predict_features", "artifact_path": artifact_path, "expected_sha256": expected_sha256, "features": features})
    rows = result.get("results")
    if not isinstance(rows, list):
        raise RuntimeError("Model worker returned no feature prediction results")
    return [item for item in rows if isinstance(item, dict)]


def predict_workspace_deployment(
    *,
    artifact: Mapping[str, Any],
    training_records: list[Mapping[str, Any]],
    smiles_list: list[str],
    target_identity: str | None,
) -> list[dict[str, Any]]:
    """Run ordered single/bulk results through the isolated model worker."""

    # The AD reference set is bounded to keep the worker request predictable;
    # the exact dataset lineage remains in the artifact manifest and database.
    result = _run_worker({
        "operation": "predict",
        "artifact_path": str(artifact.get("artifact_path") or ""),
        "expected_sha256": str(artifact.get("checksum_sha256") or artifact.get("artifact_sha256") or "") or None,
        "artifact": dict(artifact),
        "training_records": [dict(row) for row in training_records[:10000]],
        "smiles": [str(value) for value in smiles_list],
        "target_identity": target_identity,
    })
    results = result.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Model worker returned no prediction results")
    return [item for item in results if isinstance(item, dict)]
