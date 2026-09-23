"""Short-lived model worker for untrusted PKL/joblib artifacts.

The Flask process never deserializes a user-supplied artifact. The worker reads
one bounded JSON request, loads one owner-scoped file, performs one operation,
then exits. Production should run this module in the dedicated no-network
worker container defined by the deployment configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


def _check_path(path_value: str, expected_sha256: str | None = None) -> Path:
    path = Path(path_value).resolve()
    root = Path(os.environ.get("QSARIFY_MODEL_ARTIFACT_ROOT", Path(__file__).resolve().parents[1] / "models")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Model artifact is outside the configured artifact root") from exc
    if not path.is_file():
        raise FileNotFoundError("Model artifact is unavailable")
    if path.stat().st_size <= 0 or path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Model artifact exceeds the 512 MB worker limit")
    if expected_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Model artifact checksum does not match the registered checksum")
    return path


def _load(path: Path) -> Any:
    import joblib
    return joblib.load(path)


def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    path = _check_path(str(payload.get("artifact_path") or ""), str(payload.get("expected_sha256") or "") or None)
    model = _load(path)
    steps = getattr(model, "steps", None)
    classes = getattr(model, "classes_", None)
    return {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_size_bytes": path.stat().st_size,
        "model_type": type(model).__name__,
        "has_predict": bool(hasattr(model, "predict")),
        "has_predict_proba": bool(hasattr(model, "predict_proba")),
        "feature_count": int(getattr(model, "n_features_in_", 0) or 0),
        "classification_classes": [str(value) for value in (classes.tolist() if hasattr(classes, "tolist") else classes or [])],
        "pipeline_steps": [str(item[0]) for item in steps] if isinstance(steps, list) else [],
    }


def _predict(payload: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from .applicability_domain import assess_molecule_applicability_domain
    from .workspace_training import FINGERPRINT_BITS, FINGERPRINT_RADIUS, build_workspace_prediction_features

    path = _check_path(str(payload.get("artifact_path") or ""), str(payload.get("expected_sha256") or "") or None)
    model = _load(path)
    smiles_list = [str(item) for item in payload.get("smiles") or []]
    artifact = payload.get("artifact") if isinstance(payload.get("artifact"), Mapping) else {}
    training_records = [item for item in payload.get("training_records") or [] if isinstance(item, Mapping)]
    task_type = str(artifact.get("task_type") or "")
    architecture = str(artifact.get("architecture") or "")
    vocabulary = [str(value) for value in artifact.get("target_vocabulary") or [] if str(value)]
    model_card = artifact.get("model_card") if isinstance(artifact.get("model_card"), Mapping) else {}
    feature_schema = model_card.get("feature_schema") if isinstance(model_card, Mapping) else {}
    target_identity = payload.get("target_identity") if isinstance(payload.get("target_identity"), str) else None
    features, errors = build_workspace_prediction_features(smiles_list, pooled=architecture == "pooled_multitarget", target_identity=target_identity, target_vocabulary=vocabulary, feature_schema=feature_schema if isinstance(feature_schema, Mapping) else {})
    error_by_index = {int(item["index"]): str(item["error"]) for item in errors}
    fingerprints: list[list[int]] = []
    for record in training_records:
        smiles = record.get("curated_smiles") or record.get("smiles")
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
            fingerprints.append([int(bit) for bit in fp])
    training_fps = np.asarray(fingerprints, dtype=np.uint8) if fingerprints else np.empty((0, FINGERPRINT_BITS), dtype=np.uint8)
    results: list[dict[str, Any]] = []
    valid_row = 0
    for index, smiles in enumerate(smiles_list):
        if index in error_by_index:
            results.append({"index": index, "smiles": smiles, "error": error_by_index[index]})
            continue
        if valid_row >= len(features):
            results.append({"index": index, "smiles": smiles, "error": "Feature generation failed"})
            continue
        row = features[valid_row : valid_row + 1]; valid_row += 1
        molecule = Chem.MolFromSmiles(smiles)
        packed = np.packbits(training_fps, axis=1, bitorder="big") if len(training_fps) else np.empty((0, 256), dtype=np.uint8)
        ad = assess_molecule_applicability_domain(molecule, packed) if len(training_fps) else {"max_similarity": None, "status": "UNAVAILABLE"}
        prediction = model.predict(row)[0]
        result: dict[str, Any] = {"index": index, "smiles": smiles, "target_identity": target_identity, "task_type": task_type, "applicability_domain": ad}
        if task_type == "classification":
            probabilities = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
            result.update({"prediction": "Active" if int(prediction) == 1 else "Inactive", "prediction_value": int(prediction), "probability_active": float(probabilities[1]) if probabilities is not None and len(probabilities) > 1 else None, "confidence": float(np.max(probabilities)) if probabilities is not None else None})
        elif task_type == "regression":
            result.update({"prediction": float(prediction), "prediction_value": float(prediction), "confidence": None})
        else:
            result["error"] = "Unsupported deployed task type"
        results.append(result)
    return {"results": results}


def _predict_features(payload: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    path = _check_path(str(payload.get("artifact_path") or ""), str(payload.get("expected_sha256") or "") or None)
    model = _load(path)
    rows = payload.get("features")
    if not isinstance(rows, list) or not rows or len(rows) > 500:
        raise ValueError("features must contain 1-500 rows")
    matrix: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) > 10000:
            raise ValueError("Each feature row must be a bounded numeric list")
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError("Feature values must be numeric") from exc
        matrix.append(values)
    feature_count = int(getattr(model, "n_features_in_", 0) or 0)
    if feature_count and any(len(row) != feature_count for row in matrix):
        raise ValueError(f"Every feature row must contain exactly {feature_count} values")
    prediction_rows = np.asarray(matrix, dtype=float)
    predictions = model.predict(prediction_rows)
    probabilities = model.predict_proba(prediction_rows) if hasattr(model, "predict_proba") else None
    results: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions.tolist() if hasattr(predictions, "tolist") else predictions):
        row: dict[str, Any] = {"index": index, "prediction": prediction.item() if hasattr(prediction, "item") else prediction}
        if probabilities is not None:
            probability_row = probabilities[index]
            row["probabilities"] = [value.item() if hasattr(value, "item") else value for value in probability_row]
            row["confidence"] = float(np.max(probability_row))
        results.append(row)
    return {"results": results}


def main() -> int:
    raw = sys.stdin.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Worker request exceeds size limit")
    payload = json.loads(raw)
    operation = payload.get("operation") if isinstance(payload, Mapping) else None
    result = _validate(payload) if operation == "validate" else _predict(payload) if operation == "predict" else _predict_features(payload) if operation == "predict_features" else None
    if result is None:
        raise ValueError("Unsupported worker operation")
    sys.stdout.write(json.dumps(result, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(str(exc))
        raise SystemExit(1)
