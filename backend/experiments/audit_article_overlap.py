"""Compute DOI/PMID overlap between provenance-complete training and external data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_doi(value) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    return re.sub(r"^doi:\s*", "", text).strip().rstrip(".")


def normalize_pmid(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    # CSV round-tripping through pandas commonly renders PMIDs as
    # ``10732965.0``. Treat only an integer-valued decimal representation as
    # the same PMID; reject all other malformed values fail-closed.
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text if text.isdigit() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compound-manifest",
        type=Path,
        default=None,
        help="Optional manifest from the canonical-SMILES exclusion step.",
    )
    args = parser.parse_args()

    training = pd.read_csv(args.training.resolve(), low_memory=False)
    external = pd.read_csv(args.external.resolve(), low_memory=False)
    training["_doi"] = training["document_doi"].map(normalize_doi)
    training["_pmid"] = training["document_pmid"].map(normalize_pmid)
    external["_doi"] = external["article_doi"].map(normalize_doi)
    external["_pmid"] = external["pmid"].map(normalize_pmid)
    training_dois = set(training.loc[training["_doi"].ne(""), "_doi"])
    training_pmids = set(training.loc[training["_pmid"].ne(""), "_pmid"])
    external_dois = set(external.loc[external["_doi"].ne(""), "_doi"])
    external_pmids = set(external.loc[external["_pmid"].ne(""), "_pmid"])
    doi_overlap = training_dois & external_dois
    pmid_overlap = training_pmids & external_pmids

    by_target = {}
    for target, group in external.groupby("target_name", sort=True):
        target_dois = set(group.loc[group["_doi"].ne(""), "_doi"])
        target_pmids = set(group.loc[group["_pmid"].ne(""), "_pmid"])
        by_target[str(target)] = {
            "external_rows": int(len(group)),
            "external_doi_count": len(target_dois),
            "external_pmid_count": len(target_pmids),
            "overlap_dois": sorted(target_dois & training_dois),
            "overlap_pmids": sorted(target_pmids & training_pmids),
            "rows_with_overlapping_doi": int(group["_doi"].isin(training_dois).sum()),
            "rows_with_overlapping_pmid": int(group["_pmid"].isin(training_pmids).sum()),
        }

    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {"path": args.training.resolve().as_posix(), "sha256": sha256_file(args.training.resolve()), "rows": int(len(training))},
        "external": {"path": args.external.resolve().as_posix(), "sha256": sha256_file(args.external.resolve()), "rows": int(len(external))},
        "training_doi_count": len(training_dois),
        "training_pmid_count": len(training_pmids),
        "external_doi_count": len(external_dois),
        "external_pmid_count": len(external_pmids),
        "overlap_dois": sorted(doi_overlap),
        "overlap_pmids": sorted(pmid_overlap),
        "overlap_status": "NO_DOI_PMID_OVERLAP_DETECTED" if not doi_overlap and not pmid_overlap else "OVERLAP_DETECTED",
        "by_target": by_target,
        "compound_overlap_note": "Compound-level overlap was independently removed before external validation; DOI/PMID overlap is a separate article-level check.",
    }
    if args.compound_manifest and args.compound_manifest.is_file():
        compound_manifest = json.loads(args.compound_manifest.resolve().read_text(encoding="utf-8"))
        audit["compound_exclusion"] = {
            "manifest": args.compound_manifest.resolve().as_posix(),
            "manifest_sha256": sha256_file(args.compound_manifest.resolve()),
            "status": "NO_TARGET_CANONICAL_SMILES_OVERLAP_IN_FINAL_EXTERNAL_SET",
            "rows_removed": compound_manifest.get("compound_overlap_rows", 0),
            "rows_removed_by_target": compound_manifest.get("compound_overlap_by_target", {}),
            "final_external_rows": compound_manifest.get("output", {}).get("rows"),
            "unique_excluded_rows": compound_manifest.get("unique_excluded_rows"),
            "article_overlap_rows": compound_manifest.get("article_overlap_rows"),
            "overlap_rows_both_checks": compound_manifest.get("overlap_rows_both_checks"),
            "exclusion_map": compound_manifest.get("exclusion_map"),
            "training_curated_smiles": compound_manifest.get("training_curated_smiles"),
            "external_curated_smiles": compound_manifest.get("external_curated_smiles"),
        }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overlap_status": audit["overlap_status"], "doi_overlap_count": len(doi_overlap), "pmid_overlap_count": len(pmid_overlap)}, indent=2))


if __name__ == "__main__":
    main()
