"""Audit BindingDB external-validation comparability and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED_TARGETS = {"AChE": "P22303", "BACE1": "P56817", "COX-2": "P35354", "MAO-B": "P27338", "VISFATIN": "P43490"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_coverage(series: pd.Series) -> float:
    values = series.astype("string").fillna("").str.strip()
    return float(values.ne("").mean()) if len(values) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    training_path = args.training_input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    training = pd.read_csv(training_path, low_memory=False)

    required_columns = {
        "target_name",
        "uniprot_id",
        "smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "target_organism_bindingdb",
        "curation_source",
        "publication_date",
        "bindingdb_date",
        "article_doi",
        "pmid",
        "assay_description",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"External set is missing required audit fields: {missing}")

    frame["publication_date_parsed"] = pd.to_datetime(frame["publication_date"], errors="coerce")
    frame["bindingdb_date_parsed"] = pd.to_datetime(frame["bindingdb_date"], errors="coerce")
    by_target = {}
    for target, group in frame.groupby("target_name", sort=True):
        source_counts = group["curation_source"].value_counts(dropna=False).to_dict()
        class_counts = group["bioactivity_class"].value_counts(dropna=False).to_dict()
        description_text = group["assay_description"].astype("string").fillna("").str.lower()
        by_target[str(target)] = {
            "rows": int(len(group)),
            "uniprot_ids": sorted(group["uniprot_id"].dropna().astype(str).unique().tolist()),
            "target_names": sorted(group["target_name_bindingdb"].dropna().astype(str).unique().tolist()),
            "organisms": sorted(group["target_organism_bindingdb"].dropna().astype(str).unique().tolist()),
            "class_distribution": {str(key): int(value) for key, value in class_counts.items()},
            "source_distribution": {str(key): int(value) for key, value in source_counts.items()},
            "standard_type_values": sorted(group["standard_type"].dropna().astype(str).unique().tolist()),
            "standard_units_values": sorted(group["standard_units"].dropna().astype(str).unique().tolist()),
            "standard_relation_values": sorted(group["standard_relation"].dropna().astype(str).unique().tolist()),
            "assay_description_coverage": nonempty_coverage(group["assay_description"]),
            "assay_description_terms": {
                term: int(description_text.str.contains(term, regex=False).sum())
                for term in ("enzyme", "cell", "recombinant", "substrate", "inhib", "binding", "fluorescen", "radioligand")
            },
            "publication_date_min": group["publication_date_parsed"].min().date().isoformat() if group["publication_date_parsed"].notna().any() else None,
            "publication_date_max": group["publication_date_parsed"].max().date().isoformat() if group["publication_date_parsed"].notna().any() else None,
            "bindingdb_date_min": group["bindingdb_date_parsed"].min().date().isoformat() if group["bindingdb_date_parsed"].notna().any() else None,
            "bindingdb_date_max": group["bindingdb_date_parsed"].max().date().isoformat() if group["bindingdb_date_parsed"].notna().any() else None,
        }

    training_provenance_columns = [column for column in ("article_doi", "pmid", "publication_date", "assay_chembl_id", "assay_description") if column in training.columns]
    training_has_provenance = bool(training_provenance_columns)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPARABILITY_REVIEW_REQUIRED",
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path), "rows": int(len(frame))},
        "training_input": {"path": training_path.as_posix(), "sha256": sha256_file(training_path), "rows": int(len(training))},
        "target_contract": {"expected": EXPECTED_TARGETS, "observed": {str(key): sorted(value) for key, value in frame.groupby("target_name")["uniprot_id"]}},
        "global_checks": {
            "all_expected_targets_present": set(EXPECTED_TARGETS).issubset(set(frame["target_name"].unique())),
            "all_uniprot_ids_match_target_contract": all(frame.loc[frame["target_name"] == target, "uniprot_id"].eq(uniprot).all() for target, uniprot in EXPECTED_TARGETS.items()),
            "all_ic50": bool(frame["standard_type"].eq("IC50").all()),
            "all_nm": bool(frame["standard_units"].eq("nM").all()),
            "all_exact_relation": bool(frame["standard_relation"].eq("=").all()),
            "assay_description_coverage": nonempty_coverage(frame["assay_description"]),
            "training_publication_provenance_available": training_has_provenance,
            "training_provenance_columns": training_provenance_columns,
            "article_level_overlap_status": "UNKNOWN_TRAINING_SNAPSHOT_LACKS_DOI_PMID_FIELDS" if not training_has_provenance else "REQUIRES_DOI_PMID_COMPARISON",
        },
        "by_target": by_target,
        "decision": {
            "usable_as_candidate_external_evidence": True,
            "usable_as_final_oecd_predictivity_evidence": False,
            "blocking_reasons": [
                "Article-level overlap cannot be excluded from the original training snapshot because it lacks DOI/PMID provenance.",
                "Assay descriptions are present, but endpoint/assay comparability requires expert review.",
                "The external distributions are heterogeneous and must be interpreted with balanced accuracy/MCC, not ROC-AUC alone.",
            ],
        },
    }
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit["global_checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
