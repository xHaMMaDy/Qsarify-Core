"""Attach ChEMBL document DOI/PMID metadata to a rich activity snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API = "https://www.ebi.ac.uk/chembl/api/data/document.json"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_document(document_id: str) -> tuple[str, dict | None, str | None]:
    for attempt in range(3):
        try:
            response = requests.get(API, params={"document_chembl_id": document_id, "limit": 1}, timeout=60)
            response.raise_for_status()
            documents = response.json().get("documents", [])
            return document_id, documents[0] if documents else {}, None
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                return document_id, None, str(exc)
    return document_id, None, "unreachable"


def fetch_pmid_for_doi(doi: str) -> str:
    """Resolve a missing ChEMBL PMID through Europe PMC when possible."""
    if not doi or not str(doi).strip():
        return ""
    try:
        response = requests.get(
            EUROPE_PMC_API,
            params={"query": f'DOI:"{str(doi).strip()}"', "format": "json", "pageSize": 1},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        if results:
            return str(results[0].get("pmid") or "").strip()
    except (requests.RequestException, ValueError, TypeError):
        return ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    document_ids = sorted(frame["document_chembl_id"].dropna().astype(str).unique())
    metadata: dict[str, dict] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_document, document_id) for document_id in document_ids]
        for future in as_completed(futures):
            document_id, payload, error = future.result()
            if error:
                errors[document_id] = error
            else:
                metadata[document_id] = payload or {}

    fields = {
        "document_doi": "doi",
        "document_pmid": "pubmed_id",
        "document_title": "title",
        "document_year": "year",
        "document_journal": "journal",
        "document_type": "doc_type",
    }
    for output_column, source_field in fields.items():
        frame[output_column] = frame["document_chembl_id"].map(lambda value: metadata.get(str(value), {}).get(source_field, ""))
    # ChEMBL does not backfill PMID for every document even when a DOI is
    # present. Resolve only those missing values so the shipped audit is a
    # genuine DOI+PMID audit rather than a DOI-only audit with a PMID label.
    missing_pmid_dois = sorted(
        {
            str(doi).strip()
            for doi, pmid in zip(frame["document_doi"], frame["document_pmid"])
            if str(doi).strip() and not str(pmid).strip()
        }
    )
    resolved_pmids: dict[str, str] = {}
    if missing_pmid_dois:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_pmid_for_doi, doi): doi for doi in missing_pmid_dois}
            for future in as_completed(futures):
                doi = futures[future]
                resolved_pmids[doi] = future.result()
        frame["document_pmid"] = [
            str(pmid).strip() or resolved_pmids.get(str(doi).strip(), "")
            for doi, pmid in zip(frame["document_doi"], frame["document_pmid"])
        ]
    frame.to_csv(output_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path), "rows": int(len(frame))},
        "output": {"path": output_path.as_posix(), "sha256": sha256_file(output_path), "rows": int(len(frame))},
        "api": API,
        "documents_requested": len(document_ids),
        "documents_resolved": len(metadata),
        "documents_failed": len(errors),
        "failed_document_ids": errors,
        "doi_coverage": float(frame["document_doi"].astype("string").fillna("").str.strip().ne("").mean()),
        "pmid_coverage": float(frame["document_pmid"].astype("string").fillna("").str.strip().ne("").mean()),
        "pmid_resolution": {
            "resolver": EUROPE_PMC_API,
            "doi_values_queried": len(missing_pmid_dois),
            "pmids_resolved": int(sum(bool(value) for value in resolved_pmids.values())),
        },
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
