"""Run a reproducible QSARify benchmark from a labelled compound table.

Expected input columns:
    smiles, bioactivity_class

Optional categorical target columns can be included with --one-hot-column.
The output directory contains the exact configuration, input checksum,
software versions, per-model metrics, and (optionally) serialized models.
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
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

BACKEND_DIR = Path(__file__).resolve().parents[1]
if not (BACKEND_DIR / "app.py").is_file():
    BACKEND_DIR = BACKEND_DIR.parent / "source" / "backend"
PROJECT_ROOT = BACKEND_DIR.parent
if PROJECT_ROOT.name.lower() == "source":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import calculate_descriptors, get_fingerprints, get_models  # noqa: E402
from services.curation import curate_smiles  # noqa: E402
from services.evaluation import calculate_classification_metrics  # noqa: E402


DEFAULT_DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"]
DEFAULT_ACTIVITY_THRESHOLD_NM = 10000.0
DEFAULT_MODELS = [
    "LogisticRegression",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "SVC",
    "MLPClassifier",
]


def parse_smiles(value):
    """Return an RDKit molecule or None for missing/non-string SMILES."""
    curated = curate_smiles(value)
    return Chem.MolFromSmiles(curated) if curated is not None else None


def canonical_smiles(value):
    """Return a canonical SMILES string or ``None`` for an invalid value."""
    molecule = parse_smiles(value)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def prepare_benchmark_frame(
    frame: pd.DataFrame,
    *,
    one_hot_column: str | None,
    activity_threshold_nm: float = DEFAULT_ACTIVITY_THRESHOLD_NM,
) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic QSAR record cleaning and duplicate handling.

    ChEMBL activity exports can contain repeated measurements for one
    target/compound pair and occasional measurements on opposite sides of a
    binary threshold. For datasets with numeric ``standard_value`` in nM, the
    repeated measurements are represented by their median and the class is
    recomputed from the configured threshold. This avoids duplicate leakage
    and does not choose an arbitrary measurement. For generic labelled files
    without ``standard_value``, consistent duplicates are collapsed and
    conflicting groups are excluded as ambiguous.
    """
    required = {"smiles", "bioactivity_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    working = frame.copy()
    audit = {
        "input_rows": int(len(working)),
        "invalid_label_rows": 0,
        "invalid_activity_rows": 0,
        "invalid_smiles_rows": 0,
        "rows_after_activity_filter": int(len(working)),
        "rows_after_smiles_filter": 0,
        "duplicate_groups_collapsed": 0,
        "duplicate_rows_collapsed": 0,
        "conflicting_duplicate_groups": 0,
        "conflicting_duplicate_rows": 0,
        "ambiguous_groups_dropped": 0,
        "ambiguous_rows_dropped": 0,
        "duplicate_policy": "median standard_value per target/compound when available; otherwise drop conflicting groups",
        "activity_threshold_nm": float(activity_threshold_nm),
    }

    valid_labels = working["bioactivity_class"].isin(["Active", "Inactive"])
    audit["invalid_label_rows"] = int((~valid_labels).sum())
    working = working.loc[valid_labels].copy()

    if "standard_units" in working.columns:
        units = working["standard_units"].astype("string").str.strip().str.lower()
        invalid_units = units.notna() & (units != "nm")
        if invalid_units.any():
            raise ValueError("Benchmark input contains non-nM standard_units; convert units before evaluation.")

    has_numeric_activity = "standard_value" in working.columns
    if has_numeric_activity:
        activity = pd.to_numeric(working["standard_value"], errors="coerce")
        valid_activity = activity.notna() & np.isfinite(activity) & (activity > 0)
        audit["invalid_activity_rows"] = int((~valid_activity).sum())
        working = working.loc[valid_activity].copy()
        working["_activity_numeric"] = activity.loc[valid_activity].astype(float)
    audit["rows_after_activity_filter"] = int(len(working))

    working["_canonical_smiles"] = working["smiles"].map(canonical_smiles)
    valid_smiles = working["_canonical_smiles"].notna()
    audit["invalid_smiles_rows"] = int((~valid_smiles).sum())
    working = working.loc[valid_smiles].copy()
    audit["rows_after_smiles_filter"] = int(len(working))
    if working.empty:
        raise ValueError("No valid labelled molecules remain after activity and SMILES validation.")

    # Target identity is part of the biological observation, not merely a
    # feature choice. Preserve target-specific records even when a caller
    # deliberately disables target one-hot encoding for an ablation.
    dedup_column = one_hot_column
    if dedup_column is None and "target_name" in working.columns:
        dedup_column = "target_name"
    group_columns = ["_canonical_smiles"]
    if dedup_column:
        if dedup_column not in working.columns:
            raise ValueError(f"Deduplication/one-hot column '{dedup_column}' is not present in the input.")
        group_columns.insert(0, dedup_column)
    audit["dedup_group_columns"] = list(group_columns)

    working["_dedup_group"] = working.groupby(group_columns, dropna=False, sort=False).ngroup()
    group_sizes = working.groupby("_dedup_group", sort=False).size()
    label_counts = working.groupby("_dedup_group", sort=False)["bioactivity_class"].nunique(dropna=False)
    audit["duplicate_groups_collapsed"] = int((group_sizes > 1).sum())
    audit["duplicate_rows_collapsed"] = int((group_sizes - 1).clip(lower=0).sum())
    audit["conflicting_duplicate_groups"] = int((label_counts > 1).sum())
    audit["conflicting_duplicate_rows"] = int(group_sizes[label_counts > 1].sum())

    representatives = working.drop_duplicates("_dedup_group", keep="first").copy()
    if has_numeric_activity:
        medians = working.groupby("_dedup_group", sort=False)["_activity_numeric"].median()
        representatives["_activity_numeric"] = representatives["_dedup_group"].map(medians)
        representatives["standard_value"] = representatives["_activity_numeric"]
        representatives["bioactivity_class"] = np.where(
            representatives["_activity_numeric"] <= activity_threshold_nm,
            "Active",
            "Inactive",
        )
        representatives["pIC50"] = -np.log10(representatives["_activity_numeric"] * 1e-9)
    else:
        ambiguous_groups = label_counts[label_counts > 1].index
        audit["ambiguous_groups_dropped"] = int(len(ambiguous_groups))
        audit["ambiguous_rows_dropped"] = int(group_sizes.loc[ambiguous_groups].sum()) if len(ambiguous_groups) else 0
        representatives = representatives[~representatives["_dedup_group"].isin(ambiguous_groups)].copy()

    representatives["smiles"] = representatives["_canonical_smiles"]
    return representatives.drop(columns=["_canonical_smiles", "_dedup_group", "_activity_numeric"], errors="ignore"), audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_project_path(path: Path) -> str:
    """Represent a packaged project path without embedding the local checkout."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def scaffold_key(molecule) -> str:
    """Return a deterministic Bemis-Murcko scaffold key for a molecule."""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        return scaffold or "__acyclic__"
    except Exception:
        return "__invalid__"


def get_split_indices(
    index: np.ndarray,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    split_strategy: str,
    groups: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if split_strategy == "random":
        return train_test_split(index, test_size=test_size, random_state=seed, stratify=y)
    if split_strategy != "scaffold":
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    if groups is None:
        raise ValueError("Scaffold splitting requires scaffold groups.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_positions, test_positions = next(splitter.split(index, y, groups=groups.loc[index].to_numpy()))
    return index[train_positions], index[test_positions]


def build_features(
    frame: pd.DataFrame,
    *,
    fingerprint_type: str,
    fingerprint_radius: int,
    fingerprint_bits: int,
    descriptor_names: list[str],
    one_hot_column: str | None,
    split_test_size: float,
    split_seed: int,
    split_strategy: str,
    activity_threshold_nm: float = DEFAULT_ACTIVITY_THRESHOLD_NM,
    use_fingerprints: bool = True,
    use_descriptors: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    clean, data_audit = prepare_benchmark_frame(
        frame,
        one_hot_column=one_hot_column,
        activity_threshold_nm=activity_threshold_nm,
    )
    # ``prepare_benchmark_frame`` already returns curated canonical SMILES;
    # parse them directly here to avoid repeating expensive tautomer curation.
    clean["mol"] = clean["smiles"].map(
        lambda value: Chem.MolFromSmiles(value) if isinstance(value, str) else None
    )
    clean = clean[clean["mol"].notna()].copy()
    if clean.empty:
        raise ValueError("No valid labelled molecules remain after SMILES validation.")

    y = clean["bioactivity_class"].map({"Active": 1, "Inactive": 0}).astype(int)
    scaffolds = clean["mol"].map(scaffold_key)
    train_indices, _ = get_split_indices(
        clean.index.to_numpy(), y, test_size=split_test_size, seed=split_seed,
        split_strategy=split_strategy, groups=scaffolds,
    )

    feature_parts: list[pd.DataFrame] = []
    feature_metadata = {
        "fingerprint_type": fingerprint_type,
        "fingerprint_radius": fingerprint_radius,
        "fingerprint_bits": fingerprint_bits,
        "descriptor_names": descriptor_names,
        "one_hot_column": one_hot_column,
        "use_fingerprints": bool(use_fingerprints),
        "use_descriptors": bool(use_descriptors),
        "split_strategy": split_strategy,
        "data_audit": data_audit,
    }

    if use_fingerprints:
        fingerprints = clean["mol"].map(
            lambda mol: get_fingerprints(mol, fingerprint_type, fingerprint_radius, fingerprint_bits)
        )
        feature_parts.append(
            pd.DataFrame(
                [list(fp) for fp in fingerprints],
                index=clean.index,
                columns=[f"fp_{i}" for i in range(fingerprint_bits)],
            )
        )

    # Keep the feature contract identical to the built-in predictor:
    # fingerprints, target one-hot block, then descriptors.  A different
    # block order preserves dimensionality but silently changes predictions.
    if one_hot_column:
        if one_hot_column not in clean.columns:
            raise ValueError(f"One-hot column '{one_hot_column}' is not present in the input.")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_categories = clean.loc[train_indices, [one_hot_column]].fillna("<missing>")
        all_categories = clean[[one_hot_column]].fillna("<missing>")
        encoder.fit(train_categories)
        encoded = encoder.transform(all_categories)
        feature_parts.append(
            pd.DataFrame(
                encoded,
                index=clean.index,
                columns=encoder.get_feature_names_out([one_hot_column]),
            )
        )
        feature_metadata["one_hot_categories"] = [list(map(str, values)) for values in encoder.categories_]

    if use_descriptors:
        descriptors = clean["mol"].map(lambda mol: calculate_descriptors(mol, descriptor_names))
        feature_parts.append(pd.DataFrame(descriptors.tolist(), index=clean.index, columns=descriptor_names))

    if not feature_parts:
        raise ValueError("At least one feature block must be enabled.")

    X = pd.concat(feature_parts, axis=1)
    X.columns = X.columns.astype(str)
    X.attrs["scaffold_keys"] = scaffolds
    feature_metadata["feature_count"] = int(X.shape[1])
    feature_metadata["feature_order"] = []
    if use_fingerprints:
        feature_metadata["feature_order"].append("fingerprints")
    if one_hot_column:
        feature_metadata["feature_order"].append("one_hot")
    if use_descriptors:
        feature_metadata["feature_order"].append("descriptors")
    feature_metadata["row_count"] = int(X.shape[0])
    feature_metadata["class_distribution"] = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    return X, y, feature_metadata


def prepare_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    split_strategy: str,
    imputer_strategy: str,
    use_resampling: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    train_indices, test_indices = get_split_indices(
        X.index.to_numpy(), y, test_size=test_size, seed=seed,
        split_strategy=split_strategy, groups=X.attrs.get("scaffold_keys"),
    )
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    metadata = {
        "test_size": test_size,
        "seed": seed,
        "imputer_strategy": imputer_strategy,
        "resampling": bool(use_resampling),
        "split_strategy": split_strategy,
        "train_rows_before_resampling": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    if split_strategy == "scaffold":
        scaffold_keys = X.attrs["scaffold_keys"]
        train_scaffolds = set(scaffold_keys.loc[train_indices])
        test_scaffolds = set(scaffold_keys.loc[test_indices])
        metadata["train_scaffold_groups"] = len(train_scaffolds)
        metadata["test_scaffold_groups"] = len(test_scaffolds)
        metadata["scaffold_overlap_groups"] = len(train_scaffolds & test_scaffolds)

    if imputer_strategy == "drop":
        train_valid = ~X_train.isna().any(axis=1)
        test_valid = ~X_test.isna().any(axis=1)
        X_train, y_train = X_train.loc[train_valid], y_train.loc[train_valid]
        X_test, y_test = X_test.loc[test_valid], y_test.loc[test_valid]
    else:
        imputer = SimpleImputer(strategy=imputer_strategy)
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    if use_resampling and y_train.value_counts().min() > 1:
        k_neighbors = min(5, int(y_train.value_counts().min()) - 1)
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors) if k_neighbors < 5 else SMOTE(random_state=seed)
        resampler = SMOTETomek(random_state=seed, n_jobs=-1, smote=smote)
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_resampled, columns=X.columns)
        y_train = pd.Series(y_resampled, name=y.name)

    metadata["train_rows_after_resampling"] = int(len(X_train))
    metadata["train_class_distribution"] = {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()}
    metadata["test_class_distribution"] = {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()}
    return X_train, X_test, y_train, y_test, metadata


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = calculate_classification_metrics(y_test, y_pred)
    metrics["classification_report"] = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Inactive", "Active"],
            output_dict=True,
            zero_division=0,
        )
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    if probabilities is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
        metrics["pr_auc"] = float(average_precision_score(y_test, probabilities))
        precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
        metrics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
    else:
        metrics["roc_auc"] = None
        metrics["roc_curve"] = None
        metrics["pr_auc"] = None
        metrics["pr_curve"] = None
    return metrics


def run_benchmark(args: argparse.Namespace) -> dict:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Read the complete table before type inference so mixed optional metadata
    # columns do not emit chunked-parser warnings or vary with file size.
    frame = pd.read_csv(input_path, low_memory=False)
    descriptor_names = [item.strip() for item in args.descriptors.split(",") if item.strip()]
    models_requested = [item.strip() for item in args.models.split(",") if item.strip()]
    activity_threshold_nm = float(getattr(args, "activity_threshold_nm", DEFAULT_ACTIVITY_THRESHOLD_NM))
    use_fingerprints = bool(getattr(args, "use_fingerprints", True))
    use_descriptors = bool(getattr(args, "use_descriptors", True))

    X, y, feature_metadata = build_features(
        frame,
        fingerprint_type=args.fingerprint_type,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        descriptor_names=descriptor_names,
        one_hot_column=args.one_hot_column,
        split_test_size=args.test_size,
        split_seed=args.seed,
        split_strategy=args.split_strategy,
        activity_threshold_nm=activity_threshold_nm,
        use_fingerprints=use_fingerprints,
        use_descriptors=use_descriptors,
    )
    X_train, X_test, y_train, y_test, split_metadata = prepare_split(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        split_strategy=args.split_strategy,
        imputer_strategy=args.imputer_strategy,
        use_resampling=args.resampling,
    )

    registry = get_models()
    unknown = sorted(set(models_requested) - set(registry))
    if unknown:
        raise ValueError(f"Unknown or unavailable model(s): {unknown}")

    results = {}
    for model_name in models_requested:
        model = registry[model_name]
        model.fit(X_train, y_train)
        results[model_name] = evaluate_model(model, X_test, y_test)
        results[model_name]["parameters"] = model.get_params(deep=False)
        if args.save_models:
            joblib.dump(model, output_dir / f"model_{model_name}.pkl")

    majority_class = int(y_train.value_counts().sort_values(ascending=False).index[0])
    majority_predictions = np.full(len(y_test), majority_class, dtype=int)
    baselines = {
        "majority_class": {
            "class": majority_class,
            "train_active_fraction": float(np.mean(y_train.to_numpy() == 1)),
            "test_active_fraction": float(np.mean(y_test.to_numpy() == 1)),
            **calculate_classification_metrics(y_test, majority_predictions),
            "no_skill_pr_auc": float(np.mean(y_test.to_numpy() == 1)),
        }
    }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": portable_project_path(input_path),
            "sha256": sha256_file(input_path),
            "rows_read": int(len(frame)),
        },
        "configuration": {
            "models": models_requested,
            "fingerprint_type": args.fingerprint_type,
            "fingerprint_radius": args.fingerprint_radius,
            "fingerprint_bits": args.fingerprint_bits,
            "descriptors": descriptor_names,
            "one_hot_column": args.one_hot_column,
            "test_size": args.test_size,
            "seed": args.seed,
            "split_strategy": args.split_strategy,
            "imputer_strategy": args.imputer_strategy,
            "resampling": args.resampling,
            "activity_threshold_nm": activity_threshold_nm,
            "use_fingerprints": use_fingerprints,
            "use_descriptors": use_descriptors,
        },
        "feature_metadata": feature_metadata,
        "split_metadata": split_metadata,
        "baselines": baselines,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"manifest": manifest, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with smiles and bioactivity_class columns")
    parser.add_argument("--output", required=True, help="Directory for manifest, results, and optional models")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--fingerprint-type", default="Morgan", choices=["Morgan", "RDKit", "Topological"])
    parser.add_argument("--fingerprint-radius", type=int, default=3)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--descriptors", default=",".join(DEFAULT_DESCRIPTOR_NAMES))
    parser.add_argument("--one-hot-column", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["random", "scaffold"], default="random")
    parser.add_argument("--imputer-strategy", choices=["mean", "median", "most_frequent", "constant", "drop"], default="mean")
    parser.add_argument("--activity-threshold-nm", type=float, default=DEFAULT_ACTIVITY_THRESHOLD_NM)
    parser.add_argument("--no-fingerprints", dest="use_fingerprints", action="store_false")
    parser.add_argument("--no-descriptors", dest="use_descriptors", action="store_false")
    parser.set_defaults(use_fingerprints=True, use_descriptors=True)
    parser.add_argument("--resampling", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    payload = run_benchmark(cli_args)
    print(json.dumps({"output": str(Path(cli_args.output).resolve()), "models": list(payload["results"])}, indent=2))
