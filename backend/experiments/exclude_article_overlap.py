"""Remove BindingDB rows that overlap the refreshed training snapshot.

The external set is filtered at both levels required by the evidence contract:
article identifiers (DOI/PMID) and target-specific curated canonical SMILES.
The latter is intentionally recomputed from the current training snapshot so a
provenance refresh cannot silently re-introduce compound leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from audit_article_overlap import normalize_doi, normalize_pmid
from services.curation import curate_smiles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    args = parser.parse_args()
    training = pd.read_csv(args.training.resolve(), low_memory=False)
    external = pd.read_csv(args.external.resolve(), low_memory=False)
    for frame_name, frame in (("training", training), ("external", external)):
        missing = {"target_name", args.smiles_column} - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} input is missing required columns: {sorted(missing)}")

    # Keep the comparison target-specific because the frozen models are
    # target-specific.  Curation is deliberately identical to the benchmark
    # path (salt stripping, uncharging, bounded tautomer canonicalisation,
    # valence and organic-subset checks).
    training["_canonical_smiles"] = training[args.smiles_column].map(curate_smiles)
    external["_canonical_smiles"] = external[args.smiles_column].map(curate_smiles)
    external["_source_row_id"] = range(len(external))
    training_keys = set(zip(training["target_name"], training["_canonical_smiles"])) - {("", ""), (None, None)}
    training_keys = {key for key in training_keys if key[1]}
    external_keys = list(zip(external["target_name"], external["_canonical_smiles"]))
    compound_overlap = pd.Series([key in training_keys for key in external_keys], index=external.index)
    invalid_external = external["_canonical_smiles"].isna()
    training_dois = set(training["document_doi"].map(normalize_doi)) - {""}
    training_pmids = set(training["document_pmid"].map(normalize_pmid)) - {""}
    external_dois = external["article_doi"].map(normalize_doi)
    external_pmids = external["pmid"].map(normalize_pmid)
    overlap_doi = external_dois.isin(training_dois)
    overlap_pmid = external_pmids.isin(training_pmids)
    article_overlap = overlap_doi | overlap_pmid
    excluded = external.loc[article_overlap | compound_overlap | invalid_external].copy()
    excluded["exclusion_reason"] = "DOI"
    excluded.loc[overlap_pmid & ~overlap_doi, "exclusion_reason"] = "PMID"
    excluded.loc[compound_overlap & ~article_overlap, "exclusion_reason"] = "TARGET_CANONICAL_SMILES"
    excluded.loc[invalid_external & ~article_overlap & ~compound_overlap, "exclusion_reason"] = "INVALID_CURATED_SMILES"
    excluded["article_overlap_doi"] = overlap_doi.loc[excluded.index]
    excluded["article_overlap_pmid"] = overlap_pmid.loc[excluded.index]
    excluded["compound_overlap"] = compound_overlap.loc[excluded.index]
    excluded["invalid_curated_smiles"] = invalid_external.loc[excluded.index]
    excluded["exclusion_reasons"] = [
        ";".join(
            reason
            for reason, matched in (
                ("DOI", bool(overlap_doi.loc[index])),
                ("PMID", bool(overlap_pmid.loc[index])),
                ("TARGET_CANONICAL_SMILES", bool(compound_overlap.loc[index])),
                ("INVALID_CURATED_SMILES", bool(invalid_external.loc[index])),
            )
            if matched
        )
        for index in excluded.index
    ]
    exclusion_map_columns = [
        "_source_row_id", "target_name", args.smiles_column, "_canonical_smiles", "article_doi", "pmid",
        "standard_value", "bioactivity_class", "exclusion_reason", "exclusion_reasons",
        "article_overlap_doi", "article_overlap_pmid", "compound_overlap", "invalid_curated_smiles",
    ]
    map_frame = excluded[exclusion_map_columns].copy()
    map_frame = map_frame.rename(columns={"_source_row_id": "source_row_id", "_canonical_smiles": "curated_canonical_smiles"})
    kept = external.loc[~(article_overlap | compound_overlap | invalid_external)].copy()
    for frame in (excluded, kept):
        frame[args.smiles_column] = frame["_canonical_smiles"]
        frame.drop(columns=["_canonical_smiles"], inplace=True)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path = output_path.with_name(output_path.stem + "_excluded.csv")
    kept.to_csv(output_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    exclusion_map_path = output_path.with_name(output_path.stem + "_exclusion_map.csv")
    map_frame.to_csv(exclusion_map_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {"path": args.training.resolve().as_posix(), "sha256": sha256_file(args.training.resolve()), "rows": int(len(training))},
        "external_input": {"path": args.external.resolve().as_posix(), "sha256": sha256_file(args.external.resolve()), "rows": int(len(external))},
        "output": {"path": output_path.as_posix(), "sha256": sha256_file(output_path), "rows": int(len(kept))},
        "excluded_output": {"path": excluded_path.as_posix(), "sha256": sha256_file(excluded_path), "rows": int(len(excluded))},
        "exclusion_map": {"path": exclusion_map_path.as_posix(), "sha256": sha256_file(exclusion_map_path), "rows": int(len(map_frame))},
        "excluded_by_target": {str(key): int(value) for key, value in excluded["target_name"].value_counts().sort_index().items()},
        "excluded_by_reason": {str(key): int(value) for key, value in excluded["exclusion_reason"].value_counts().sort_index().items()},
        "training_curated_smiles": int(training["_canonical_smiles"].notna().sum()),
        "external_curated_smiles": int(external["_canonical_smiles"].notna().sum()),
        "compound_overlap_rows": int(compound_overlap.sum()),
        "article_overlap_rows": int(article_overlap.sum()),
        "overlap_rows_both_checks": int((article_overlap & compound_overlap).sum()),
        "unique_excluded_rows": int(len(excluded)),
        "check_level_exclusion_accounting": "article_overlap_rows + compound_overlap_rows - overlap_rows_both_checks = unique_excluded_rows",
        "invalid_external_smiles_rows": int(invalid_external.sum()),
        "compound_overlap_by_target": {
            str(key): int(value)
            for key, value in external.loc[compound_overlap, "target_name"].value_counts().sort_index().items()
        },
        "overlap_dois": sorted(set(external_dois[overlap_doi]) & training_dois),
        "overlap_pmids": sorted(set(external_pmids[overlap_pmid]) & training_pmids),
        "claim_boundary": "Target-specific curated canonical-SMILES and DOI/PMID disjointness were enforced against the current provenance-enriched ChEMBL training snapshot; endpoint/assay expert review remains required.",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
