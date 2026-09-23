"""Run a fixed QSARify benchmark over several pre-specified random seeds.

Each seed gets its own manifest/results directory. The aggregate JSON reports
mean and sample standard deviation for the scalar metrics; it is intended for
uncertainty-aware software-paper evidence, not for selecting a model after
seeing the test results.
"""

from __future__ import annotations

import argparse
import json
import statistics
from argparse import Namespace
from pathlib import Path

from run_benchmark import DEFAULT_ACTIVITY_THRESHOLD_NM, DEFAULT_DESCRIPTOR_NAMES, DEFAULT_MODELS, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="13,42,77")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--fingerprint-type", default="Morgan", choices=["Morgan", "RDKit", "Topological"])
    parser.add_argument("--fingerprint-radius", type=int, default=3)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--descriptors", default=",".join(DEFAULT_DESCRIPTOR_NAMES))
    parser.add_argument("--one-hot-column", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--imputer-strategy", choices=["mean", "median", "most_frequent", "constant", "drop"], default="mean")
    parser.add_argument("--activity-threshold-nm", type=float, default=DEFAULT_ACTIVITY_THRESHOLD_NM)
    parser.add_argument("--resampling", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in seeds:
        run_args = Namespace(**vars(args))
        run_args.seed = seed
        run_args.output = str(output_root / f"seed-{seed}")
        runs[str(seed)] = run_benchmark(run_args)

    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    aggregate = {}
    for model_name in model_names:
        aggregate[model_name] = {}
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"):
            values = [runs[str(seed)]["results"][model_name][metric] for seed in seeds]
            aggregate[model_name][metric] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }

    baseline_metrics = {}
    baseline_runs = [runs[str(seed)]["manifest"]["baselines"]["majority_class"] for seed in seeds]
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "no_skill_pr_auc"):
        values = [float(item[metric]) for item in baseline_runs]
        baseline_metrics[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    baseline_metrics["class"] = baseline_runs[0]["class"]
    baseline_metrics["train_active_fraction"] = {
        "mean": statistics.fmean(float(item["train_active_fraction"]) for item in baseline_runs),
        "sample_std": statistics.stdev(float(item["train_active_fraction"]) for item in baseline_runs) if len(baseline_runs) > 1 else 0.0,
    }

    summary = {
        "seeds": seeds,
        "split_strategy": args.split_strategy,
        "models": model_names,
        "configuration": {
            "fingerprint_type": args.fingerprint_type,
            "fingerprint_radius": args.fingerprint_radius,
            "fingerprint_bits": args.fingerprint_bits,
            "descriptors": [item.strip() for item in args.descriptors.split(",") if item.strip()],
            "one_hot_column": args.one_hot_column,
            "test_size": args.test_size,
            "imputer_strategy": args.imputer_strategy,
            "resampling": args.resampling,
            "activity_threshold_nm": args.activity_threshold_nm,
        },
        "runs": {seed: {"manifest": payload["manifest"], "results_path": f"seed-{seed}/results.json"} for seed, payload in runs.items()},
        "aggregate": aggregate,
        "baselines": {"majority_class": baseline_metrics},
    }
    (output_root / "repeated_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    cli_args = parse_args()
    payload = main(cli_args)
    print(json.dumps({"output": str(Path(cli_args.output).resolve()), "seeds": payload["seeds"]}, indent=2))
