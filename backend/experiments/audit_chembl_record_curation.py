"""Build a bounded, reproducible ChEMBL record-level curation manifest."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000


def fetch_records(target_id: str) -> list[dict]:
    records: list[dict] = []
    offset = 0
    while True:
        response = requests.get(
            f"{API}/activity.json",
            params={
                "target_chembl_id": target_id,
                "standard_type": "IC50",
                "standard_units": "nM",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("activities") or []
        records.extend(batch)
        total = int((payload.get("page_meta") or {}).get("total_count") or 0)
        if not batch or len(records) >= total:
            return records
        offset += len(batch)


def summary(rows: list[dict]) -> dict:
    relation = Counter(str(r.get("standard_relation") or r.get("relation") or "missing") for r in rows)
    assay_type = Counter(str(r.get("assay_type") or "missing") for r in rows)
    bao_format = Counter(str(r.get("bao_format") or "missing") for r in rows)
    species = Counter(str(r.get("target_organism") or "missing") for r in rows)
    flags = Counter(str(r.get("standard_flag")) for r in rows)
    validity = Counter(str(r.get("data_validity_comment") or "none") for r in rows)
    structures = sum(bool(r.get("canonical_smiles")) for r in rows)
    numeric = 0
    for row in rows:
        try:
            value = float(row.get("standard_value"))
            numeric += value > 0
        except (TypeError, ValueError):
            pass
    keys = [(r.get("molecule_chembl_id"), r.get("target_chembl_id"), r.get("standard_type")) for r in rows]
    duplicate_keys = sum(count - 1 for count in Counter(keys).values() if count > 1)
    return {
        "retrieved_records": len(rows),
        "numeric_positive_standard_value": numeric,
        "standard_units_nM": sum(r.get("standard_units") == "nM" for r in rows),
        "canonical_smiles_present": structures,
        "canonical_smiles_missing": len(rows) - structures,
        "relation_counts": dict(relation),
        "assay_type_counts": dict(assay_type),
        "bao_format_counts": dict(bao_format),
        "species_counts": dict(species),
        "standard_flag_counts": dict(flags),
        "data_validity_comment_counts": dict(validity),
        "potential_duplicate_records": sum(str(r.get("potential_duplicate")) == "1" for r in rows),
        "duplicate_compound_target_endpoint_rows": duplicate_keys,
    }


def curate(rows: list[dict]) -> dict:
    """Apply the frozen equality-only policy and retain auditable row IDs."""
    retained = []
    excluded = Counter()
    excluded_ids: dict[str, list[int | str]] = {}
    for row in rows:
        record_id = row.get("activity_id") or row.get("record_id") or row.get("molecule_chembl_id")
        reasons = []
        try:
            value = float(row.get("standard_value"))
            if value <= 0:
                reasons.append("non_positive_value")
        except (TypeError, ValueError):
            reasons.append("non_numeric_value")
        if row.get("standard_units") != "nM":
            reasons.append("unit_not_nM")
        if row.get("standard_relation") not in ("=", None) or row.get("relation") not in ("=", None):
            reasons.append("non_equality_relation")
        if str(row.get("standard_flag")) != "1":
            reasons.append("standard_flag_0")
        if str(row.get("potential_duplicate")) == "1":
            reasons.append("potential_duplicate")
        if row.get("data_validity_comment"):
            reasons.append("data_validity_comment")
        if not row.get("canonical_smiles"):
            reasons.append("missing_canonical_smiles")
        if row.get("target_organism") != "Homo sapiens":
            reasons.append("non_human_target")
        if reasons:
            for reason in reasons:
                excluded[reason] += 1
                excluded_ids.setdefault(reason, []).append(record_id)
        else:
            retained.append(row)
    groups: dict[tuple, list[dict]] = {}
    for row in retained:
        key = (row.get("molecule_chembl_id"), row.get("target_chembl_id"), row.get("standard_type"))
        groups.setdefault(key, []).append(row)
    aggregated = []
    for key, group in groups.items():
        values = sorted(float(r["standard_value"]) for r in group)
        median = values[len(values) // 2] if len(values) % 2 else (values[len(values)//2 - 1] + values[len(values)//2]) / 2
        base = group[0]
        aggregated.append({"molecule_chembl_id": key[0], "target_chembl_id": key[1], "standard_type": key[2], "standard_units": "nM", "standard_value_median": median, "source_record_count": len(group), "activity_ids": [r.get("activity_id") for r in group]})
    return {"retained_pre_aggregation": len(retained), "retained_unique_compound_target_endpoint": len(aggregated), "excluded_by_reason": dict(excluded), "excluded_record_ids_by_reason": excluded_ids, "aggregated_records": aggregated}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    mapping_path = root / "docs" / "target-intelligence" / "benchmark_v1_modality_mapping.csv"
    output_path = root / "docs" / "target-intelligence" / "CHEMBL_RECORD_CURATION_MANIFEST_2026-09-17.json"
    mappings = list(csv.DictReader(mapping_path.open(encoding="utf-8")))
    entries = []
    for mapping in mappings:
        if mapping["modality_stratum"] != "small_molecule":
            continue
        target_rows = requests.get(
            f"{API}/target.json",
            params={"target_components__accession": mapping["uniprot_accession"], "limit": 20},
            timeout=60,
        ).json().get("targets") or []
        target = next((t for t in target_rows if t.get("organism") == "Homo sapiens" and t.get("target_type") == "SINGLE PROTEIN"), None)
        if not target:
            raise RuntimeError(f"No human single-protein mapping for {mapping['target_gene']}")
        records = fetch_records(target["target_chembl_id"])
        item = {
            "gene": mapping["target_gene"],
            "uniprot_accession": mapping["uniprot_accession"],
            "chembl_target_id": target["target_chembl_id"],
            "collection_policy": {"standard_type": "IC50", "standard_units": "nM"},
            "summary": summary(records),
            "curation": curate(records),
        }
        entries.append(item)
        print(f"{item['gene']}: {item['summary']['retrieved_records']} records", flush=True)
    manifest = {
        "manifest_version": "ti-chembl-record-curation-2026-09-17",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base": API,
        "source_mapping": str(mapping_path.relative_to(root)).replace("\\", "/"),
        "policy": {"standard_type": "IC50", "standard_units": "nM", "human_single_protein_only": True, "relation": "=", "standard_flag": "1", "potential_duplicate": "0", "require_canonical_smiles": True, "aggregate_key": ["molecule_chembl_id", "target_chembl_id", "standard_type"], "aggregate_value": "median"},
        "targets": entries,
        "limitations": [
            "This manifest summarizes retrieved activity records and does not establish biological validity or OECD validation.",
            "Duplicate aggregation and final structure/value exclusions must be applied by the training dataset builder using these summaries and the frozen policy.",
        ],
    }
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
