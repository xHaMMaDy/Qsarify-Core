"""Real chemical curation for saved Target Modeling Workspace datasets."""

from __future__ import annotations

from typing import Any, Mapping

from services.curation import curate_smiles


def curate_dataset(records: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    curated: list[dict[str, Any]] = []
    excluded = 0
    missing_smiles = 0
    for row in records:
        smiles_value = row.get("smiles") or row.get("canonical_smiles")
        if not isinstance(smiles_value, str) or not smiles_value.strip():
            missing_smiles += 1
            continue
        canonical = curate_smiles(smiles_value)
        if canonical is None:
            excluded += 1
            continue
        item = dict(row)
        item["original_smiles"] = smiles_value
        item["smiles"] = canonical
        item["curated_smiles"] = canonical
        curated.append(item)
    return curated, {"input_records": len(records), "curated_records": len(curated), "excluded_invalid": excluded, "excluded_missing_smiles": missing_smiles}
