"""Deterministic feature preparation for curated workspace datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold(molecule) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or "__acyclic__"
    except Exception:
        return "__invalid__"


def prepare_workspace_dataset(records: list[Mapping[str, Any]], *, task_type: str, pooled: bool = False) -> dict[str, Any]:
    if task_type not in {"regression", "classification"}:
        raise ValueError("task_type must be regression or classification")
    if not records:
        raise ValueError("No curated records were supplied")
    prepared = []
    excluded = 0
    for row in records:
        smiles = row.get("curated_smiles") or row.get("smiles")
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            excluded += 1
            continue
        if task_type == "regression":
            value = row.get("pIC50")
            try:
                label = float(value)
            except (TypeError, ValueError):
                excluded += 1
                continue
        else:
            label_value = str(row.get("bioactivity_class") or "")
            if label_value not in {"Active", "Inactive"}:
                excluded += 1
                continue
            label = 1 if label_value == "Active" else 0
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=3, nBits=2048)
        target_identity = str(row.get("target_identity") or row.get("target_key") or "")
        prepared.append({
            "features": list(fp),
            "descriptors": {
                "MolWt": float(Descriptors.MolWt(molecule)),
                "MolLogP": float(Crippen.MolLogP(molecule)),
                "NumHDonors": int(Lipinski.NumHDonors(molecule)),
                "NumHAcceptors": int(Lipinski.NumHAcceptors(molecule)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(molecule)),
            },
            "label": label,
            "scaffold": _scaffold(molecule),
            "target_identity": target_identity if pooled else None,
            "molecule_chembl_id": row.get("molecule_chembl_id"),
        })
    if not prepared:
        raise ValueError("No valid labelled records remain after feature preparation")
    labels = [item["label"] for item in prepared]
    summary = {
        "input_records": len(records),
        "prepared_records": len(prepared),
        "excluded_records": excluded,
        "task_type": task_type,
        "pooled": bool(pooled),
        "feature_width": 2048 + 5 + (len({item["target_identity"] for item in prepared}) if pooled else 0),
        "scaffold_count": len({item["scaffold"] for item in prepared}),
        "label_counts": {str(key): int(value) for key, value in Counter(labels).items()},
        "target_counts": {str(key): int(value) for key, value in Counter(item["target_identity"] for item in prepared if item["target_identity"]).items()},
        "split_strategy": "scaffold_disjoint",
        "fingerprint": {"type": "Morgan", "radius": 3, "bits": 2048},
        "descriptor_names": ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"],
    }
    return {"summary": summary, "prepared_records": prepared}
