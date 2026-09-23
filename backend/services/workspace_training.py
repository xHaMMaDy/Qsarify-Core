"""Deterministic scaffold-disjoint QSAR training for workspace datasets."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - deployment capability dependent
    xgb = None

MODEL_NAMES = {"linear_regression", "logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting", "support_vector_machine", "neural_network"} | ({"xgboost"} if xgb is not None else set())
DESCRIPTOR_NAMES = ["NumAtoms", "AtomicNumberSum", "NumHeavyAtoms", "NumBonds", "NumRings"]
FINGERPRINT_RADIUS = 3
FINGERPRINT_BITS = 2048


def _features(records: list[Mapping[str, Any]], *, pooled: bool) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    rows: list[list[float]] = []
    labels: list[Any] = []
    scaffolds: list[str] = []
    targets: list[str] = []
    target_vocabulary = sorted({str(row.get("target_identity") or row.get("target_key") or "") for row in records if pooled})
    for row in records:
        smiles = row.get("curated_smiles") or row.get("smiles")
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            continue
        if "label" in row:
            label = row["label"]
        elif row.get("bioactivity_class") in {"Active", "Inactive"}:
            label = 1 if row["bioactivity_class"] == "Active" else 0
        else:
            label = row.get("pIC50")
        if label is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
        descriptor_values = [float(mol.GetNumAtoms()), float(sum(atom.GetAtomicNum() for atom in mol.GetAtoms())), float(mol.GetNumHeavyAtoms()), float(mol.GetNumBonds()), float(mol.GetRingInfo().NumRings())]
        one_hot = [1.0 if str(row.get("target_identity") or row.get("target_key") or "") == target else 0.0 for target in target_vocabulary] if pooled else []
        rows.append([float(bit) for bit in fp] + one_hot + descriptor_values)
        labels.append(label)
        scaffold_value = row.get("scaffold")
        if not scaffold_value:
            try:
                scaffold_value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "__acyclic__"
            except Exception:
                scaffold_value = "__invalid__"
        scaffolds.append(str(scaffold_value))
        targets.append(str(row.get("target_identity") or row.get("target_key") or ""))
    if not rows:
        raise ValueError("No valid prepared records remain for training")
    return np.asarray(rows, dtype=np.float32), labels, np.asarray(scaffolds), np.asarray(targets), target_vocabulary


def build_workspace_prediction_features(
    smiles_list: list[str],
    *,
    pooled: bool,
    target_identity: str | None,
    target_vocabulary: list[str],
    feature_schema: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the exact feature layout used by workspace training.

    The returned errors are per-input and deliberately preserve input order so
    bulk prediction can report invalid structures without shifting rows. The
    feature schema is checked instead of silently padding/truncating a model's
    inputs, which prevents an apparently successful but scientifically invalid
    deployment.
    """

    schema = feature_schema or {}
    fingerprint = schema.get("fingerprint") if isinstance(schema, Mapping) else None
    radius = int((fingerprint or {}).get("radius", FINGERPRINT_RADIUS)) if isinstance(fingerprint, Mapping) else FINGERPRINT_RADIUS
    bits = int((fingerprint or {}).get("bits", FINGERPRINT_BITS)) if isinstance(fingerprint, Mapping) else FINGERPRINT_BITS
    descriptor_names = list(schema.get("descriptor_names") or DESCRIPTOR_NAMES) if isinstance(schema, Mapping) else list(DESCRIPTOR_NAMES)
    if radius != FINGERPRINT_RADIUS or bits != FINGERPRINT_BITS or descriptor_names != DESCRIPTOR_NAMES:
        raise ValueError("Unsupported workspace artifact feature schema")
    vocabulary = [str(value) for value in target_vocabulary if str(value)]
    if pooled and not vocabulary:
        raise ValueError("Pooled deployment has no frozen target vocabulary")
    if pooled and not target_identity:
        raise ValueError("target_identity is required for pooled prediction")
    if pooled and str(target_identity) not in vocabulary:
        raise ValueError(f"Unknown target identity '{target_identity}' for this pooled model")

    rows: list[list[float]] = []
    errors: list[dict[str, Any]] = []
    for index, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            errors.append({"index": index, "error": "Invalid SMILES string"})
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=radius, nBits=bits)
        descriptor_values = [
            float(molecule.GetNumAtoms()),
            float(sum(atom.GetAtomicNum() for atom in molecule.GetAtoms())),
            float(molecule.GetNumHeavyAtoms()),
            float(molecule.GetNumBonds()),
            float(molecule.GetRingInfo().NumRings()),
        ]
        one_hot = [1.0 if str(target_identity) == target else 0.0 for target in vocabulary] if pooled else []
        rows.append([float(bit) for bit in fp] + one_hot + descriptor_values)
    if not rows:
        return np.empty((0, bits + len(vocabulary) + len(descriptor_names))), errors
    return np.asarray(rows, dtype=np.float32), errors


def _models(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        models = {
            "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13))]),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=13, n_jobs=-1, class_weight="balanced"),
            "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=13, n_jobs=-1, class_weight="balanced"),
            "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, random_state=13),
            "support_vector_machine": SVC(probability=True, class_weight="balanced", random_state=13),
            "neural_network": MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True, random_state=13),
        }
        if xgb is not None:
            models["xgboost"] = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=13, n_jobs=-1)
        return models
    models = {
        "linear_regression": Pipeline([("scale", StandardScaler(with_mean=False)), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=13, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=13, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=150, random_state=13),
        "support_vector_machine": SVR(C=1.0, epsilon=0.1),
        "neural_network": MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True, random_state=13),
    }
    if xgb is not None:
        models["xgboost"] = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=13, n_jobs=-1)
    return models


def train_workspace_dataset(records: list[Mapping[str, Any]], *, task_type: str, architecture: str, model_names: list[str], workspace_id: str, artifact_root: str, owner_id: str | None = None) -> dict[str, Any]:
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression")
    if architecture not in {"separate_models", "pooled_multitarget"}:
        raise ValueError("Training requires a confirmed separate or pooled architecture")
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError("Unsupported model family")
    X, labels, groups, targets, target_vocabulary = _features(records, pooled=architecture == "pooled_multitarget")
    y = np.asarray(labels, dtype=float if task_type == "regression" else int)
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two scaffold groups are required for scaffold-disjoint training")
    train_idx = test_idx = None
    for seed in (13, 42, 77):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        candidate_train, candidate_test = next(splitter.split(X, y, groups=groups))
        if task_type != "classification" or (len(np.unique(y[candidate_train])) >= 2 and len(np.unique(y[candidate_test])) >= 2):
            train_idx, test_idx = candidate_train, candidate_test
            break
    if train_idx is None or test_idx is None:
        raise ValueError("No scaffold-disjoint split contains both classes")
    owner_folder = f"user_{uuid.UUID(owner_id)}" if owner_id else "legacy_workspace"
    artifact_dir = Path(artifact_root) / owner_folder / "trained-models" / f"study_{uuid.UUID(workspace_id)}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name in dict.fromkeys(model_names):
        estimator = _models(task_type).get(name)
        if estimator is None:
            continue
        train_features = X[train_idx]
        test_features = X[test_idx]
        if name == "linear_regression":
            train_features = sparse.csr_matrix(train_features, dtype=np.float32)
            test_features = sparse.csr_matrix(test_features, dtype=np.float32)
        estimator.fit(train_features, y[train_idx])
        prediction = estimator.predict(test_features)
        metrics: dict[str, float]
        if task_type == "classification":
            probabilities = estimator.predict_proba(X[test_idx])[:, 1] if hasattr(estimator, "predict_proba") else None
            metrics = {
                "accuracy": float(accuracy_score(y[test_idx], prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], prediction)),
                "precision": float(precision_score(y[test_idx], prediction, zero_division=0)),
                "recall": float(recall_score(y[test_idx], prediction, zero_division=0)),
                "f1": float(f1_score(y[test_idx], prediction, zero_division=0)),
                "mcc": float(matthews_corrcoef(y[test_idx], prediction)),
                "roc_auc": float(roc_auc_score(y[test_idx], probabilities)) if probabilities is not None else 0.0,
                "pr_auc": float(average_precision_score(y[test_idx], probabilities)) if probabilities is not None else 0.0,
            }
            selection_score = metrics["mcc"]
        else:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
                "mae": float(mean_absolute_error(y[test_idx], prediction)),
                "r2": float(r2_score(y[test_idx], prediction)) if len(test_idx) > 1 else 0.0,
            }
            selection_score = metrics["r2"]
        fingerprint_matrix = X[train_idx, :2048].astype(np.uint8)
        query_fps = X[test_idx, :2048].astype(np.uint8)
        similarities = []
        for query in query_fps:
            q = [int(value) for value in query]
            similarities.append(max(DataStructs.TanimotoSimilarity(DataStructs.CreateFromBitString("".join(map(str, q))), DataStructs.CreateFromBitString("".join(map(str, row)))) for row in fingerprint_matrix))
        artifact_name = f"{name}-{task_type}-{uuid.uuid4().hex}.joblib"
        artifact_path = artifact_dir / artifact_name
        joblib.dump(estimator, artifact_path)
        checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        manifest = {"schema_version": "workspace-model-v2", "artifact_format": "joblib", "model_file": artifact_name, "model_sha256": checksum, "model_size_bytes": artifact_path.stat().st_size, "task_type": task_type, "architecture": architecture, "model_family": name, "feature_schema": {"fingerprint": {"radius": FINGERPRINT_RADIUS, "bits": FINGERPRINT_BITS}, "descriptor_names": DESCRIPTOR_NAMES}, "target_vocabulary": target_vocabulary, "training_records": int(len(train_idx)), "test_records": int(len(test_idx)), "scaffold_count": int(len(np.unique(groups)))}
        manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_manifest.replace(manifest_path)
        results[name] = {"metrics": metrics, "selection_score": selection_score, "artifact_path": str(artifact_path), "artifact_sha256": checksum, "artifact_size_bytes": artifact_path.stat().st_size, "manifest_path": str(manifest_path), "manifest": manifest, "ad": {"test_compounds": len(similarities), "mean_max_similarity": float(np.mean(similarities)) if similarities else 0.0, "out_of_domain_fraction": float(sum(value < 0.5 for value in similarities) / len(similarities)) if similarities else 1.0}}
    if not results:
        raise ValueError("No model candidates were trained")
    best = max(results, key=lambda key: results[key]["selection_score"])
    return {"task_type": task_type, "architecture": architecture, "target_vocabulary": target_vocabulary, "train_records": int(len(train_idx)), "test_records": int(len(test_idx)), "scaffold_count": int(len(np.unique(groups))), "candidates": results, "best_model": best, "feature_schema": {"fingerprint": {"radius": 3, "bits": 2048}, "descriptor_names": DESCRIPTOR_NAMES}}
