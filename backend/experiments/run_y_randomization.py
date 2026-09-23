"""Run a fixed y-randomization chance-correlation check.

Labels are permuted globally so the pooled class count is preserved while
compound-to-label and target-prevalence relationships are destroyed. Target
one-hot encoding is disabled in the diagnostic. The same scaffold split,
molecular feature contract, and Random Forest configuration are then
evaluated. This is an internal chance-correlation diagnostic, not external
validation or model selection evidence.
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

BACKEND_DIR = Path(__file__).resolve().parents[1]
if not (BACKEND_DIR / "app.py").is_file():
    BACKEND_DIR = BACKEND_DIR.parent / "source" / "backend"
PROJECT_ROOT = BACKEND_DIR.parent
if PROJECT_ROOT.name.lower() == "source":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import get_models  # noqa: E402
from experiments.run_benchmark import (  # noqa: E402
    DEFAULT_ACTIVITY_THRESHOLD_NM,
    DEFAULT_DESCRIPTOR_NAMES,
    build_features,
    evaluate_model,
    portable_project_path,
    prepare_benchmark_frame,
    prepare_split,
    sha256_file,
)


METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc")


def permute_labels(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a copy with pooled labels shuffled while preserving class count."""
    if "target_name" not in frame.columns:
        raise ValueError("Y-randomization requires a target_name column")
    randomized = frame.copy()
    rng = np.random.default_rng(seed)
    labels = randomized["bioactivity_class"].to_numpy(copy=True)
    rng.shuffle(labels)
    randomized["bioactivity_class"] = labels
    return randomized


def run_single(input_path: Path, output_dir: Path, seed: int, n_estimators: int) -> dict:
    frame = pd.read_csv(input_path, low_memory=False)
    clean, original_audit = prepare_benchmark_frame(
        frame,
        one_hot_column=None,
        activity_threshold_nm=DEFAULT_ACTIVITY_THRESHOLD_NM,
    )
    randomized = permute_labels(
        clean[["smiles", "bioactivity_class", "target_name"]], seed
    )
    features, labels, feature_metadata = build_features(
        randomized,
        fingerprint_type="Morgan",
        fingerprint_radius=3,
        fingerprint_bits=2048,
        descriptor_names=DEFAULT_DESCRIPTOR_NAMES,
        one_hot_column=None,
        split_test_size=0.2,
        split_seed=seed,
        split_strategy="scaffold",
        activity_threshold_nm=DEFAULT_ACTIVITY_THRESHOLD_NM,
    )
    X_train, X_test, y_train, y_test, split_metadata = prepare_split(
        features,
        labels,
        test_size=0.2,
        seed=seed,
        split_strategy="scaffold",
        imputer_strategy="mean",
        use_resampling=False,
    )
    model = get_models()["RandomForestClassifier"]
    model.set_params(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    return {
        "configuration": {
            "model": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "seed": seed,
            "split_strategy": "scaffold",
            "test_size": 0.2,
            "one_hot_column": None,
            "activity_threshold_nm": DEFAULT_ACTIVITY_THRESHOLD_NM,
            "imputer_strategy": "mean",
            "resampling": False,
            "label_permutation": "global pooled permutation; pooled class count preserved; target encoding disabled",
        },
        "input": {"path": portable_project_path(input_path), "sha256": sha256_file(input_path), "rows_read": len(frame)},
        "original_data_audit": original_audit,
        "feature_metadata": feature_metadata,
        "split_metadata": split_metadata,
        "results": results,
    }


def main(args: argparse.Namespace) -> dict:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    started = time.perf_counter()
    runs = {}
    for seed in seeds:
        seed_dir = output_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        payload = run_single(input_path, output_dir, seed, args.n_estimators)
        (seed_dir / "results.json").write_text(json.dumps(payload["results"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "purpose": "Global y-randomization molecular-feature chance-correlation check with target encoding disabled; not external validation or model selection.",
            **payload,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
        }
        (seed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        runs[str(seed)] = manifest

    summary = {
        "purpose": "Global y-randomization molecular-feature chance-correlation check with target encoding disabled; not external validation or model selection.",
        "seeds": seeds,
        "model": "RandomForestClassifier",
        "configuration": {"n_estimators": args.n_estimators, "split_strategy": "scaffold", "one_hot_column": None},
        "runs": {seed: {metric: payload["results"][metric] for metric in METRICS} for seed, payload in runs.items()},
        "aggregate": {
            metric: {
                "mean": float(np.mean([runs[str(seed)]["results"][metric] for seed in seeds])),
                "sample_std": float(np.std([runs[str(seed)]["results"][metric] for seed in seeds], ddof=1)) if len(seeds) > 1 else 0.0,
            }
            for metric in METRICS
        },
        "input": {
            "path": portable_project_path(input_path),
            "sha256": sha256_file(input_path),
            "rows_read": max(sum(1 for _ in input_path.open(encoding="utf-8")) - 1, 0),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *METRICS])
        writer.writeheader()
        for seed in seeds:
            writer.writerow({"seed": seed, **{metric: runs[str(seed)]["results"][metric] for metric in METRICS}})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="13,42,77")
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), indent=2, sort_keys=True))
