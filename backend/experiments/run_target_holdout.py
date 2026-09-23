"""Run a leave-one-target-out cross-target stress analysis.

This is a deliberately conservative supplementary analysis. Each target is
held out in full, the model is trained on the other targets, and target one-hot
encoding is disabled so the test measures molecular-feature transfer rather
than memorization of a target category. The result is not external validation
and is not used for model selection or the headline benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if not (BACKEND_DIR / "app.py").is_file():
    BACKEND_DIR = BACKEND_DIR.parent / "source" / "backend"
PROJECT_ROOT = BACKEND_DIR.parent
if PROJECT_ROOT.name.lower() == "source":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import get_models  # noqa: E402
from services.evaluation import calculate_classification_metrics  # noqa: E402
from experiments.run_benchmark import (  # noqa: E402
    DEFAULT_ACTIVITY_THRESHOLD_NM,
    DEFAULT_DESCRIPTOR_NAMES,
    build_features,
    portable_project_path,
    prepare_benchmark_frame,
    sha256_file,
)


def run_target_holdout(input_path: Path, output_dir: Path, seed: int, n_estimators: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = pd.read_csv(input_path, low_memory=False)
    clean, audit = prepare_benchmark_frame(
        frame,
        one_hot_column="target_name",
        activity_threshold_nm=DEFAULT_ACTIVITY_THRESHOLD_NM,
    )
    if "target_name" not in clean.columns:
        raise ValueError("Target holdout requires a target_name column")

    # No target one-hot block is used in this stress analysis. The helper still
    # supplies the canonical Morgan+descriptor feature construction and audit.
    features, labels, feature_metadata = build_features(
        frame,
        fingerprint_type="Morgan",
        fingerprint_radius=3,
        fingerprint_bits=2048,
        descriptor_names=DEFAULT_DESCRIPTOR_NAMES,
        one_hot_column=None,
        split_test_size=0.2,
        split_seed=seed,
        split_strategy="random",
        activity_threshold_nm=DEFAULT_ACTIVITY_THRESHOLD_NM,
    )
    targets = clean.loc[features.index, "target_name"].astype(str)
    results: list[dict] = []
    model_registry = get_models()
    if "RandomForestClassifier" not in model_registry:
        raise ValueError("RandomForestClassifier is unavailable")

    for held_out_target in sorted(targets.unique()):
        train_mask = targets != held_out_target
        test_mask = targets == held_out_target
        X_train, X_test = features.loc[train_mask], features.loc[test_mask]
        y_train, y_test = labels.loc[train_mask], labels.loc[test_mask]
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=features.columns, index=X_test.index)

        model = model_registry["RandomForestClassifier"]
        model.set_params(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "held_out_target": held_out_target,
                "train_targets": sorted(targets.loc[train_mask].unique().tolist()),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_active_fraction": float(y_train.mean()),
                "test_active_fraction": float(y_test.mean()),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "pr_auc": float(average_precision_score(y_test, probabilities)),
                **calculate_classification_metrics(y_test, predicted),
            }
        )

    summary_path = output_dir / "target_holdout_summary.csv"
    fieldnames = list(results[0])
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "Leave-one-target-out cross-target stress analysis; not external validation, model selection, or headline performance evidence.",
        "input": {
            "path": portable_project_path(input_path),
            "sha256": sha256_file(input_path),
            "rows_read": int(len(frame)),
        },
        "configuration": {
            "model": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "seed": seed,
            "test_definition": "all records for one target held out; remaining four targets used for training",
            "one_hot_column": None,
            "fingerprint_type": "Morgan",
            "fingerprint_radius": 3,
            "fingerprint_bits": 2048,
            "descriptors": DEFAULT_DESCRIPTOR_NAMES,
            "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM,
            "imputer_strategy": "mean fitted on each target-specific training partition",
        },
        "data_audit": audit,
        "feature_metadata": {**feature_metadata, "one_hot_column": None, "feature_count": int(features.shape[1])},
        "targets": sorted(targets.unique().tolist()),
        "results": results,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "outputs": {"summary": summary_path.name},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_target_holdout(Path(args.input).resolve(), Path(args.output).resolve(), args.seed, args.n_estimators), indent=2))
