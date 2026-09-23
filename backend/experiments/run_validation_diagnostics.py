"""Run fixed-protocol diagnostics for the QSARify case-study benchmark.

This script is deliberately separate from the primary benchmark.  It fits one
pre-specified Random Forest on the seed-42 scaffold split, then records
descriptive per-target metrics, uncalibrated probability diagnostics, and
nearest-training-compound Tanimoto similarity.  It does not tune a model,
choose a threshold, or replace the repeated scaffold result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import DataStructs, rdBase

BACKEND_DIR = Path(__file__).resolve().parents[1]
if not (BACKEND_DIR / "app.py").is_file():
    BACKEND_DIR = BACKEND_DIR.parent / "source" / "backend"
PROJECT_ROOT = BACKEND_DIR.parent
if PROJECT_ROOT.name.lower() == "source":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import get_fingerprints, get_models  # noqa: E402
from services.evaluation import calculate_classification_metrics  # noqa: E402
from services.applicability_domain import similarity_status  # noqa: E402
from experiments.run_benchmark import (  # noqa: E402
    DEFAULT_ACTIVITY_THRESHOLD_NM,
    build_features,
    parse_smiles,
    portable_project_path,
    prepare_benchmark_frame,
    prepare_split,
    sha256_file,
)
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def _metric(value):
    return None if value is None else float(value)


def classification_metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    """Return metrics for a group without failing on a single-class subset."""
    result = {
        "n": int(len(y_true)),
        "active": int(np.sum(y_true.to_numpy() == 1)),
        "inactive": int(np.sum(y_true.to_numpy() == 0)),
        **calculate_classification_metrics(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result.update(
            {
                "roc_auc": _metric(roc_auc_score(y_true, probabilities)),
                "pr_auc": _metric(average_precision_score(y_true, probabilities)),
                "brier_score": _metric(brier_score_loss(y_true, probabilities)),
                "log_loss": _metric(log_loss(y_true, probabilities, labels=[0, 1])),
            }
        )
    else:
        result.update({"roc_auc": None, "pr_auc": None, "brier_score": None, "log_loss": None})
    return result


def bootstrap_metric_intervals(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    iterations: int = 400,
) -> dict[str, list[float]]:
    """Return deterministic descriptive percentile intervals for held-out groups."""

    y_values = y_true.to_numpy(dtype=int)
    prediction_values = np.asarray(predictions, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    if len(y_values) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in ("accuracy", "balanced_accuracy", "mcc", "roc_auc", "pr_auc")}
    for _ in range(iterations):
        indices = rng.integers(0, len(y_values), size=len(y_values))
        sample_y = pd.Series(y_values[indices])
        sample_predictions = prediction_values[indices]
        sample_probabilities = probability_values[indices]
        point = calculate_classification_metrics(sample_y, sample_predictions)
        for metric in ("accuracy", "balanced_accuracy", "mcc"):
            samples[metric].append(float(point[metric]))
        if len(np.unique(sample_y)) == 2:
            samples["roc_auc"].append(float(roc_auc_score(sample_y, sample_probabilities)))
            samples["pr_auc"].append(float(average_precision_score(sample_y, sample_probabilities)))
    return {
        metric: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
        for metric, values in samples.items()
        if values
    }


def calibration_bins(y_true: pd.Series, probabilities: np.ndarray, bin_count: int = 10) -> tuple[list[dict], float]:
    """Calculate fixed-width calibration bins and expected calibration error."""
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bins: list[dict] = []
    weighted_error = 0.0
    total = len(probabilities)
    y_values = y_true.to_numpy(dtype=int)
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if not count:
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_fraction = float(y_values[mask].mean())
        weighted_error += (count / total) * abs(mean_probability - observed_fraction)
        bins.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": mean_probability,
                "observed_active_fraction": observed_fraction,
            }
        )
    return bins, float(weighted_error)


def nearest_training_similarity(train_molecules: list, test_molecules: list, radius: int, bits: int) -> np.ndarray:
    """Return each test molecule's maximum Morgan similarity to training data."""
    train_fingerprints = [get_fingerprints(molecule, "Morgan", radius, bits) for molecule in train_molecules]
    test_fingerprints = [get_fingerprints(molecule, "Morgan", radius, bits) for molecule in test_molecules]
    similarities = []
    for fingerprint in test_fingerprints:
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints)
        similarities.append(max(values) if values else float("nan"))
    return np.asarray(similarities, dtype=float)


def run_diagnostics(args: argparse.Namespace) -> dict:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resampling:
        raise ValueError("Diagnostics require --resampling disabled so test-row provenance is preserved.")

    started = time.perf_counter()
    frame = pd.read_csv(input_path, low_memory=False)
    clean, audit = prepare_benchmark_frame(
        frame,
        one_hot_column=args.one_hot_column,
        activity_threshold_nm=args.activity_threshold_nm,
    )
    clean["mol"] = clean["smiles"].map(parse_smiles)
    clean = clean[clean["mol"].notna()].copy()

    feature_started = time.perf_counter()
    X, y, feature_metadata = build_features(
        frame,
        fingerprint_type=args.fingerprint_type,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        descriptor_names=[item.strip() for item in args.descriptors.split(",") if item.strip()],
        one_hot_column=args.one_hot_column,
        split_test_size=args.test_size,
        split_seed=args.seed,
        split_strategy=args.split_strategy,
        activity_threshold_nm=args.activity_threshold_nm,
    )
    X_train, X_test, y_train, y_test, split_metadata = prepare_split(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        split_strategy=args.split_strategy,
        imputer_strategy=args.imputer_strategy,
        use_resampling=False,
    )
    feature_seconds = time.perf_counter() - feature_started

    models = get_models()
    if args.model not in models:
        raise ValueError(f"Unknown or unavailable model: {args.model}")
    model = models[args.model]
    train_started = time.perf_counter()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    train_seconds = time.perf_counter() - train_started

    test_rows = clean.loc[X_test.index].copy()
    test_rows["y_true"] = y_test.to_numpy()
    test_rows["predicted_class"] = predictions
    test_rows["probability_active"] = probabilities

    if args.one_hot_column and args.one_hot_column in test_rows.columns:
        per_target = {}
        for target_index, (target, group) in enumerate(test_rows.groupby(args.one_hot_column, sort=True)):
            metrics = classification_metrics(
                group["y_true"],
                group["predicted_class"].to_numpy(),
                group["probability_active"].to_numpy(),
            )
            metrics["bootstrap_95_ci"] = bootstrap_metric_intervals(
                group["y_true"],
                group["predicted_class"].to_numpy(),
                group["probability_active"].to_numpy(),
                seed=args.seed + target_index,
            )
            per_target[str(target)] = metrics
    else:
        per_target = {}

    calibration, ece = calibration_bins(y_test, probabilities, args.calibration_bins)
    ad_started = time.perf_counter()
    similarity = nearest_training_similarity(
        clean.loc[X_train.index, "mol"].tolist(),
        test_rows["mol"].tolist(),
        args.fingerprint_radius,
        args.fingerprint_bits,
    )
    ad_seconds = time.perf_counter() - ad_started
    test_rows["max_train_tanimoto"] = similarity

    # Compute target-specific AD evidence so the pooled similarity diagnostic
    # cannot be mistaken for a biologically scoped applicability domain.
    target_similarity = pd.Series(np.nan, index=test_rows.index, dtype=float)
    target_reference_counts = {}
    training_rows = clean.loc[X_train.index]
    for target, group in test_rows.groupby(args.one_hot_column, sort=True):
        target_training = training_rows[training_rows[args.one_hot_column] == target]
        if target_training.empty:
            continue
        values = nearest_training_similarity(
            target_training["mol"].tolist(),
            group["mol"].tolist(),
            args.fingerprint_radius,
            args.fingerprint_bits,
        )
        target_similarity.loc[group.index] = values
        target_reference_counts[str(target)] = int(len(target_training))
    test_rows["target_train_tanimoto"] = target_similarity
    test_rows["target_ad_status"] = target_similarity.map(
        lambda value: similarity_status(float(value)) if pd.notna(value) else "UNAVAILABLE"
    )

    target_ad_summary = {}
    ad_performance_by_status = {}
    for target, group in test_rows.groupby(args.one_hot_column, sort=True):
        status_counts = group["target_ad_status"].value_counts().to_dict()
        target_ad_summary[str(target)] = {
            "n": int(len(group)),
            "training_reference_n": target_reference_counts.get(str(target), 0),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "coverage_in_domain": float((group["target_ad_status"] == "IN_DOMAIN").mean()),
            "coverage_in_or_borderline": float(group["target_ad_status"].isin(["IN_DOMAIN", "BORDERLINE"]).mean()),
            "median_max_similarity": float(group["target_train_tanimoto"].median()),
        }
        for status, status_group in group.groupby("target_ad_status", sort=True):
            ad_performance_by_status[f"{target}:{status}"] = classification_metrics(
                status_group["y_true"],
                status_group["predicted_class"].to_numpy(),
                status_group["probability_active"].to_numpy(),
            )

    similarity_summary = {
        "n": int(len(similarity)),
        "min": float(np.nanmin(similarity)),
        "q10": float(np.nanquantile(similarity, 0.10)),
        "q25": float(np.nanquantile(similarity, 0.25)),
        "median": float(np.nanmedian(similarity)),
        "q75": float(np.nanquantile(similarity, 0.75)),
        "q90": float(np.nanquantile(similarity, 0.90)),
        "max": float(np.nanmax(similarity)),
    }

    predictions_path = output_dir / "seed-42-randomforest-test-predictions.csv"
    columns = [
        column
        for column in [args.one_hot_column, "smiles", "bioactivity_class", "y_true", "predicted_class", "probability_active", "max_train_tanimoto"]
        if column and column in test_rows.columns
    ]
    test_rows[columns].to_csv(predictions_path, index=False)

    with (output_dir / "per_target_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(per_target, handle, indent=2, sort_keys=True)
    with (output_dir / "calibration_bins.json").open("w", encoding="utf-8") as handle:
        json.dump({"bins": calibration, "expected_calibration_error": ece}, handle, indent=2, sort_keys=True)
    with (output_dir / "applicability_domain_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "reference_scope": "pooled_and_target_specific",
                "pooled": similarity_summary,
                "target_specific": target_ad_summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with (output_dir / "applicability_domain_performance.json").open("w", encoding="utf-8") as handle:
        json.dump(ad_performance_by_status, handle, indent=2, sort_keys=True)

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "Descriptive held-out diagnostics; not model selection or threshold tuning.",
        "input": {"path": portable_project_path(input_path), "sha256": sha256_file(input_path), "rows_read": int(len(frame))},
        "configuration": {
            "model": args.model,
            "fingerprint_type": args.fingerprint_type,
            "fingerprint_radius": args.fingerprint_radius,
            "fingerprint_bits": args.fingerprint_bits,
            "descriptors": [item.strip() for item in args.descriptors.split(",") if item.strip()],
            "one_hot_column": args.one_hot_column,
            "test_size": args.test_size,
            "seed": args.seed,
            "split_strategy": args.split_strategy,
            "imputer_strategy": args.imputer_strategy,
            "resampling": False,
            "activity_threshold_nm": args.activity_threshold_nm,
            "calibration_bins": args.calibration_bins,
        },
        "data_audit": audit,
        "feature_metadata": feature_metadata,
        "split_metadata": split_metadata,
        "held_out_metrics": classification_metrics(y_test, predictions, probabilities),
        "calibration": {"expected_calibration_error": ece, "bin_count": len(calibration)},
        "applicability_domain": {
            "similarity_definition": "maximum Morgan/Tanimoto similarity from each test compound to any training compound",
            "pooled": similarity_summary,
            "target_specific": target_ad_summary,
            "performance_by_target_and_status": "applicability_domain_performance.json",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "runtime_seconds": {
            "feature_and_split": float(feature_seconds),
            "fit_and_predict": float(train_seconds),
            "nearest_neighbor_similarity": float(ad_seconds),
            "total": float(time.perf_counter() - started),
        },
        "outputs": {
            "predictions": predictions_path.name,
            "per_target_metrics": "per_target_metrics.json",
            "calibration": "calibration_bins.json",
            "applicability_domain": "applicability_domain_summary.json",
            "applicability_domain_performance": "applicability_domain_performance.json",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="RandomForestClassifier")
    parser.add_argument("--fingerprint-type", default="Morgan", choices=["Morgan", "RDKit", "Topological"])
    parser.add_argument("--fingerprint-radius", type=int, default=3)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--descriptors", default="MolWt,MolLogP,NumHDonors,NumHAcceptors,TPSA")
    parser.add_argument("--one-hot-column", default="target_name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--imputer-strategy", choices=["mean", "median", "most_frequent", "constant", "drop"], default="mean")
    parser.add_argument("--activity-threshold-nm", type=float, default=DEFAULT_ACTIVITY_THRESHOLD_NM)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--resampling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    payload = run_diagnostics(cli_args)
    print(json.dumps({"output": str(Path(cli_args.output).resolve()), "model": payload["configuration"]["model"]}, indent=2))
