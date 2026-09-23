"""Run frozen target-specific OECD-oriented model evidence.

The runner trains one Random Forest per target with no target one-hot feature,
uses a scaffold-disjoint internal split, and optionally evaluates an external
target-matched CSV without refitting preprocessing. Without ``--external-input``
the output is explicitly marked ``NOT_VALIDATED``.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
from rdkit import rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

from run_benchmark import (
    DEFAULT_ACTIVITY_THRESHOLD_NM,
    DEFAULT_DESCRIPTOR_NAMES,
    build_features,
    evaluate_model,
    get_split_indices,
    sha256_file,
)
from services.applicability_domain import max_tanimoto_similarity, pack_fingerprint_matrix, similarity_status
from services.evaluation import calculate_classification_metrics


TARGETS = ["AChE", "BACE1", "COX-2", "MAO-B", "VISFATIN"]
MECHANISTIC_CONTEXT = {
    "AChE": "Acetylcholinesterase hydrolyses acetylcholine and has catalytic/peripheral anionic-site context; the model endpoint is an activity classification, not a mechanistic assay model.",
    "BACE1": "Beta-secretase 1 cleaves amyloid precursor protein and has an aspartyl catalytic dyad and flap region; the model endpoint does not establish substrate-specific or pathway mechanism.",
    "COX-2": "Cyclooxygenase-2 has a catalytic channel and a side-pocket context involving residues such as Arg120/Tyr355; assay context and inhibitor mechanism are not harmonised here.",
    "MAO-B": "Monoamine oxidase B contains an FAD-dependent active site and aromatic-cage context; the model endpoint does not distinguish reversible from mechanism-based inhibition.",
    "VISFATIN": "Visfatin/NAMPT participates in NAD biosynthesis and has a nicotinamide-pocket context; the model endpoint does not establish cellular pathway activity or mechanism.",
}


def bootstrap_confidence_intervals(y_true, y_pred, scores, *, seed: int, replicates: int = 2000) -> dict[str, list[float]]:
    """Return deterministic percentile bootstrap intervals for external metrics."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(scores, dtype=float) if scores is not None else None
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(replicates):
        indices = rng.integers(0, len(truth), size=len(truth))
        sampled_truth = truth[indices]
        sampled_predicted = predicted[indices]
        sample_metrics = calculate_classification_metrics(sampled_truth, sampled_predicted)
        if probabilities is not None and len(np.unique(sampled_truth)) == 2:
            sampled_scores = probabilities[indices]
            sample_metrics["roc_auc"] = float(roc_auc_score(sampled_truth, sampled_scores))
            sample_metrics["pr_auc"] = float(average_precision_score(sampled_truth, sampled_scores))
        for name, value in sample_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.setdefault(name, []).append(float(value))
    return {
        name: [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        for name, samples in values.items()
        if samples
    }


def fit_target_model(frame: pd.DataFrame, target: str, seed: int, threshold_nm: float) -> tuple[dict, object, SimpleImputer, np.ndarray]:
    target_frame = frame.loc[frame["target_name"] == target].copy()
    X, y, feature_metadata = build_features(
        target_frame,
        fingerprint_type="Morgan",
        fingerprint_radius=3,
        fingerprint_bits=2048,
        descriptor_names=DEFAULT_DESCRIPTOR_NAMES,
        one_hot_column=None,
        split_test_size=0.2,
        split_seed=seed,
        split_strategy="scaffold",
        activity_threshold_nm=threshold_nm,
    )
    train_indices, test_indices = get_split_indices(
        X.index.to_numpy(),
        y,
        test_size=0.2,
        seed=seed,
        split_strategy="scaffold",
        groups=X.attrs.get("scaffold_keys"),
    )
    X_train_raw, X_test_raw = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns, index=X_test_raw.index)
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics.pop("classification_report", None)
    evidence = {
        "target": target,
        "model": "RandomForestClassifier",
        "seed": seed,
        "split_strategy": "scaffold",
        "scaffold_overlap_groups": 0,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(feature_metadata["feature_count"]),
        "feature_order": feature_metadata["feature_order"],
        "metrics": metrics,
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "mechanistic_interpretation": {
            "status": "PRELIMINARY",
            "target_context": MECHANISTIC_CONTEXT[target],
            "descriptor_feature_importance": {
                name: float(value) for name, value in zip(DEFAULT_DESCRIPTOR_NAMES, model.feature_importances_[-len(DEFAULT_DESCRIPTOR_NAMES):])
            },
            "limitation": "Feature importance is descriptive and does not prove a causal or mechanistic relationship.",
        },
    }
    training_fingerprints = pack_fingerprint_matrix(X_train_raw.iloc[:, :2048].to_numpy(dtype=np.uint8))
    return evidence, model, imputer, training_fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-threshold-nm", type=float, default=DEFAULT_ACTIVITY_THRESHOLD_NM)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    external = pd.read_csv(args.external_input.resolve(), low_memory=False) if args.external_input and args.external_input.is_file() else None

    profiles = {}
    for target in TARGETS:
        evidence, model, imputer, training_fingerprints = fit_target_model(frame, target, args.seed, args.activity_threshold_nm)
        if external is not None:
            external_target = external.loc[external["target_name"] == target].copy()
            if not external_target.empty:
                external_X, external_y, _ = build_features(
                    external_target,
                    fingerprint_type="Morgan",
                    fingerprint_radius=3,
                    fingerprint_bits=2048,
                    descriptor_names=DEFAULT_DESCRIPTOR_NAMES,
                    one_hot_column=None,
                    split_test_size=0.2,
                    split_seed=args.seed,
                    split_strategy="random",
                    activity_threshold_nm=args.activity_threshold_nm,
                )
                external_features = pd.DataFrame(imputer.transform(external_X), columns=external_X.columns, index=external_X.index)
                external_metrics = evaluate_model(
                    model,
                    external_features,
                    external_y,
                )
                external_metrics.pop("classification_report", None)
                external_predictions = model.predict(external_features)
                external_scores = model.predict_proba(external_features)[:, 1] if hasattr(model, "predict_proba") else None
                external_metrics["prevalence"] = float(np.mean(external_y.to_numpy(dtype=int) == 1))
                if int((external_y == 0).sum()) == 0:
                    external_metrics["specificity"] = None
                external_metrics["bootstrap_confidence_intervals"] = bootstrap_confidence_intervals(
                    external_y.to_numpy(dtype=int),
                    external_predictions,
                    external_scores,
                    seed=args.seed + TARGETS.index(target) * 1000,
                )
                similarity = np.asarray(
                    [
                        max_tanimoto_similarity(row[:2048], training_fingerprints)
                        for row in external_X.to_numpy(dtype=np.uint8)
                    ],
                    dtype=float,
                )
                statuses = np.asarray([similarity_status(value) for value in similarity], dtype=object)
                ad_by_status = {}
                for status in ("IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN"):
                    mask = statuses == status
                    if not mask.any():
                        continue
                    metrics = calculate_classification_metrics(external_y.loc[mask], external_predictions[mask])
                    ad_by_status[status] = {
                        "n": int(mask.sum()),
                        "coverage": float(mask.mean()),
                        "median_max_similarity": float(np.median(similarity[mask])),
                        "metrics": metrics,
                    }
                evidence["external_validation"] = {
                    "status": (
                        "COMPLETED"
                        if len(external_y) >= 30 and external_y.value_counts().min() >= 10
                        else "INSUFFICIENT_CLASS_COVERAGE"
                    ),
                    "rows": int(len(external_y)),
                    "class_distribution": {str(key): int(value) for key, value in external_y.value_counts().sort_index().items()},
                    "metrics": external_metrics,
                    "applicability_domain": {
                        "thresholds": {"IN_DOMAIN": ">=0.50", "BORDERLINE": "0.30-<0.50", "OUT_OF_DOMAIN": "<0.30"},
                        "coverage": {status: int((statuses == status).sum()) / len(statuses) for status in ("IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN")},
                        "median_max_similarity": float(np.median(similarity)),
                        "performance_by_status": ad_by_status,
                    },
                }
            else:
                evidence["external_validation"] = {"status": "MISSING_TARGET_ROWS"}
        else:
            evidence["external_validation"] = {
                "status": "MISSING_EXTERNAL_DATASET",
                "message": "Internal scaffold evidence is not external validation.",
            }
        profiles[target] = evidence

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "EXTERNAL_VALIDATION_COMPLETE"
            if external is not None
            and all(profile.get("external_validation", {}).get("status") == "COMPLETED" for profile in profiles.values())
            else "EXTERNAL_VALIDATION_PARTIAL"
            if external is not None
            else "NOT_VALIDATED"
        ),
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path), "rows": int(len(frame))},
        "external_input": {"path": args.external_input.resolve().as_posix(), "sha256": sha256_file(args.external_input.resolve())} if external is not None else None,
        "configuration": {
            "targets": TARGETS,
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "seed": args.seed,
            "activity_threshold_nm": args.activity_threshold_nm,
            "fingerprint": "Morgan radius 3, 2048 bits",
            "descriptors": DEFAULT_DESCRIPTOR_NAMES,
            "one_hot_column": None,
            "external_bootstrap": {"replicates": 2000, "interval": "percentile 95%", "seed_offset_by_target": 1000},
        },
        "profiles": profiles,
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion},
    }
    (output_dir / "target_model_evidence.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "status": payload["status"], "targets": TARGETS}, indent=2))


if __name__ == "__main__":
    main()
