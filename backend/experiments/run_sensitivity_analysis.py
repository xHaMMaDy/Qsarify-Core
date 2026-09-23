"""Run pre-specified single-seed sensitivity analyses for the case study.

The primary paper result remains the repeated scaffold benchmark. This package
answers methodological questions on the same seed-42 scaffold split: activity
thresholds, feature-block ablations, target one-hot encoding, and training-only
SMOTE-Tomek resampling. No variant is used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase

BACKEND_DIR = Path(__file__).resolve().parents[1]
if not (BACKEND_DIR / "app.py").is_file():
    BACKEND_DIR = BACKEND_DIR.parent / "source" / "backend"
PROJECT_ROOT = BACKEND_DIR.parent
if PROJECT_ROOT.name.lower() == "source":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BACKEND_DIR))

from experiments.run_benchmark import (  # noqa: E402
    DEFAULT_DESCRIPTOR_NAMES,
    DEFAULT_ACTIVITY_THRESHOLD_NM,
    run_benchmark,
    sha256_file,
    portable_project_path,
)


def run_sensitivity(input_path: Path, output_dir: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        {"name": "threshold_5000_nm", "activity_threshold_nm": 5000.0, "one_hot_column": "target_name", "resampling": False, "use_fingerprints": True, "use_descriptors": True},
        {"name": "threshold_10000_nm_primary", "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM, "one_hot_column": "target_name", "resampling": False, "use_fingerprints": True, "use_descriptors": True},
        {"name": "threshold_25000_nm", "activity_threshold_nm": 25000.0, "one_hot_column": "target_name", "resampling": False, "use_fingerprints": True, "use_descriptors": True},
        {"name": "fingerprints_only", "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM, "one_hot_column": "target_name", "resampling": False, "use_fingerprints": True, "use_descriptors": False},
        {"name": "descriptors_only", "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM, "one_hot_column": "target_name", "resampling": False, "use_fingerprints": False, "use_descriptors": True},
        {"name": "without_target_one_hot", "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM, "one_hot_column": None, "resampling": False, "use_fingerprints": True, "use_descriptors": True},
        {"name": "smote_tomek_training_only", "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM, "one_hot_column": "target_name", "resampling": True, "use_fingerprints": True, "use_descriptors": True},
    ]
    rows = []
    for variant in variants:
        variant_output = output_dir / variant["name"]
        args = SimpleNamespace(
            input=str(input_path),
            output=str(variant_output),
            models="RandomForestClassifier",
            fingerprint_type="Morgan",
            fingerprint_radius=3,
            fingerprint_bits=2048,
            descriptors=",".join(DEFAULT_DESCRIPTOR_NAMES),
            one_hot_column=variant["one_hot_column"],
            test_size=0.2,
            seed=seed,
            split_strategy="scaffold",
            imputer_strategy="mean",
            activity_threshold_nm=variant["activity_threshold_nm"],
            resampling=variant["resampling"],
            use_fingerprints=variant["use_fingerprints"],
            use_descriptors=variant["use_descriptors"],
            save_models=False,
        )
        payload = run_benchmark(args)
        metrics = payload["results"]["RandomForestClassifier"]
        feature = payload["manifest"]["feature_metadata"]
        split = payload["manifest"]["split_metadata"]
        rows.append(
            {
                "variant": variant["name"],
                "activity_threshold_nm": variant["activity_threshold_nm"],
                "one_hot_column": variant["one_hot_column"] or "none",
                "resampling": variant["resampling"],
                "use_fingerprints": variant["use_fingerprints"],
                "use_descriptors": variant["use_descriptors"],
                "feature_count": feature["feature_count"],
                "model_rows": feature["row_count"],
                "test_rows": split["test_rows"],
                "scaffold_overlap_groups": split.get("scaffold_overlap_groups", 0),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "mcc": metrics["mcc"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "output": variant_output.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    summary_path = output_dir / "sensitivity_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "Pre-specified seed-42 scaffold sensitivity analyses; no variant is used for model selection.",
        "input": {"path": portable_project_path(input_path), "sha256": sha256_file(input_path), "rows_read": int(sum(1 for _ in input_path.open(encoding="utf-8")) - 1)},
        "configuration": {
            "seed": seed,
            "test_size": 0.2,
            "split_strategy": "scaffold",
            "model": "RandomForestClassifier",
            "fingerprint_type": "Morgan",
            "fingerprint_radius": 3,
            "fingerprint_bits": 2048,
            "descriptors": DEFAULT_DESCRIPTOR_NAMES,
            "imputer_strategy": "mean",
        },
        "variants": [variant["name"] for variant in variants],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion},
        "outputs": {"summary": summary_path.name},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    payload = run_sensitivity(Path(cli_args.input).resolve(), Path(cli_args.output).resolve(), cli_args.seed)
    print(json.dumps({"output": str(Path(cli_args.output).resolve()), "variants": payload["variants"]}, indent=2))
