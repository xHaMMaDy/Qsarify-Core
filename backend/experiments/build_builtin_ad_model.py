"""Build the trusted built-in AD screening model from a versioned dataset.

The generated joblib payload is intentionally separate from arbitrary
user-uploaded model loading.  It contains the estimator and the descriptor
scaler required by ``backend.app.smiles_to_features`` plus a provenance
metadata JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from rdkit import rdBase

SCRIPT_DIR = Path(__file__).resolve().parent
_source_backend = SCRIPT_DIR.parent
if not (_source_backend / "app.py").is_file():
    _source_backend = SCRIPT_DIR.parent / "source" / "backend"
BACKEND_DIR = _source_backend.resolve()
PROJECT_ROOT = BACKEND_DIR.parent
if PROJECT_ROOT.name.lower() == "source":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import PROTEIN_MAP, DESCRIPTOR_NAMES  # noqa: E402
from services.applicability_domain import pack_fingerprint_matrix, unpacked_bit_counts  # noqa: E402
from experiments.run_benchmark import (  # noqa: E402
    DEFAULT_ACTIVITY_THRESHOLD_NM,
    build_features,
    prepare_benchmark_frame,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_project_path(path: Path) -> str:
    """Return a repository-relative path so metadata remains portable."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--activity-threshold-nm", type=float, default=DEFAULT_ACTIVITY_THRESHOLD_NM)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_model = args.output_model.resolve()
    output_metadata = args.output_metadata.resolve()
    frame = pd.read_csv(input_path, low_memory=False)
    X, y, feature_metadata = build_features(
        frame,
        fingerprint_type="Morgan",
        fingerprint_radius=3,
        fingerprint_bits=2048,
        descriptor_names=DESCRIPTOR_NAMES,
        one_hot_column="target_name",
        split_test_size=0.2,
        split_seed=args.seed,
        split_strategy="random",
        activity_threshold_nm=args.activity_threshold_nm,
    )

    expected_targets = sorted(PROTEIN_MAP)
    observed_targets = feature_metadata.get("one_hot_categories", [[]])[0]
    if observed_targets != expected_targets:
        raise ValueError(f"Target category order mismatch: {observed_targets!r} != {expected_targets!r}")
    if feature_metadata.get("feature_order") != ["fingerprints", "one_hot", "descriptors"]:
        raise ValueError(f"Unexpected feature order: {feature_metadata.get('feature_order')!r}")
    if X[DESCRIPTOR_NAMES].isna().any().any():
        raise ValueError("Descriptor matrix contains missing values; refusing to build an untracked model.")
    feature_metadata["split_strategy"] = "full_fit"
    feature_metadata["encoder_fit_scope"] = "training partition used to verify category coverage"

    scaler = StandardScaler()
    # Fingerprint and one-hot blocks are integer-valued, but descriptor
    # standardisation produces floats; cast the complete matrix explicitly so
    # the persisted feature contract is stable across pandas versions.
    X_scaled = X.astype(float).copy()
    X_scaled.loc[:, DESCRIPTOR_NAMES] = scaler.fit_transform(X[DESCRIPTOR_NAMES])
    fingerprint_bits = int(feature_metadata["fingerprint_bits"])
    fingerprint_matrix = X.iloc[:, :fingerprint_bits].to_numpy(dtype=np.uint8)
    packed_training_fingerprints = pack_fingerprint_matrix(fingerprint_matrix)
    training_fingerprint_bit_counts = unpacked_bit_counts(packed_training_fingerprints)

    # Preserve target-specific reference sets for biologically scoped AD
    # reporting. The pooled matrix remains for backwards compatibility, while
    # prediction selects the requested target's training compounds whenever the
    # target is present in the artifact.
    clean, _ = prepare_benchmark_frame(
        frame,
        one_hot_column="target_name",
        activity_threshold_nm=args.activity_threshold_nm,
    )
    clean_indices = X.index.intersection(clean.index)
    if len(clean_indices) != len(X):
        raise ValueError("Curated feature rows and target reference rows are misaligned.")
    target_fingerprints = {}
    target_fingerprint_bit_counts = {}
    target_fingerprint_metadata = {}
    for target_name, group in clean.loc[clean_indices].groupby("target_name", sort=True):
        indices = group.index
        target_packed = pack_fingerprint_matrix(X.loc[indices].iloc[:, :fingerprint_bits].to_numpy(dtype=np.uint8))
        target_counts = unpacked_bit_counts(target_packed)
        target_key = str(target_name)
        target_fingerprints[target_key] = target_packed
        target_fingerprint_bit_counts[target_key] = target_counts
        target_fingerprint_metadata[target_key] = {
            "row_count": int(target_packed.shape[0]),
            "packed_width": int(target_packed.shape[1]),
            "bits": fingerprint_bits,
            "sha256": hashlib.sha256(target_packed.tobytes()).hexdigest(),
            "bit_count_min": int(target_counts.min()),
            "bit_count_max": int(target_counts.max()),
        }
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    payload = {
        "model": model,
        "scaler": scaler,
        "training_fingerprints": packed_training_fingerprints,
        "training_fingerprint_bit_counts": training_fingerprint_bit_counts,
        "training_fingerprints_by_target": target_fingerprints,
        "training_fingerprint_bit_counts_by_target": target_fingerprint_bit_counts,
        "feature_metadata": {
            **feature_metadata,
            "descriptor_scaler": "StandardScaler",
            "target_name_order": expected_targets,
            "target_accession_order": [PROTEIN_MAP[name] for name in expected_targets],
        },
        "training_metadata": {
            "fit_scope": "full_deduplicated_positive_activity_case_study_table",
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "activity_threshold_nm": args.activity_threshold_nm,
            "row_count": int(len(X_scaled)),
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "input_sha256": sha256_file(input_path),
            "training_fingerprint_count": int(packed_training_fingerprints.shape[0]),
            "training_fingerprint_packed_width": int(packed_training_fingerprints.shape[1]),
            "training_fingerprint_bits": fingerprint_bits,
            "training_fingerprint_sha256": hashlib.sha256(packed_training_fingerprints.tobytes()).hexdigest(),
            "training_fingerprint_bit_count_min": int(training_fingerprint_bit_counts.min()),
            "training_fingerprint_bit_count_max": int(training_fingerprint_bit_counts.max()),
            "training_fingerprint_by_target": target_fingerprint_metadata,
        },
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_model, compress=3)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": portable_project_path(input_path), "sha256": sha256_file(input_path), "rows_read": int(len(frame))},
        "model": {
            "path": portable_project_path(output_model),
            "sha256": sha256_file(output_model),
            "estimator": type(model).__name__,
            "classes": [int(value) for value in model.classes_],
            "n_features_in": int(model.n_features_in_),
        },
        "feature_metadata": payload["feature_metadata"],
        "training_metadata": payload["training_metadata"],
        "software": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "scikit_learn": sklearn.__version__,
        },
    }
    output_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
