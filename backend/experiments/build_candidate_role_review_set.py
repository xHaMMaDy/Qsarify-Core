"""Build an auditable candidate-role review set from frozen reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cases = {r["case_id"]: r for r in csv.DictReader((root / "docs/target-intelligence/benchmark_cases.v1-frozen.csv").open(encoding="utf-8-sig"))}
    rows = []
    for line in (root / "docs/target-intelligence/benchmark_reports.v1-frozen-source.jsonl").open(encoding="utf-8"):
        item = json.loads(line); case = cases[item["case_id"]]
        gold = set(x.strip() for x in case["expected_uniprot_accessions"].split("|") if x.strip())
        for target in item["report"].get("targets") or []:
            identifiers = target.get("identifiers") or {}
            accession = identifiers.get("uniprot") or ""
            rows.append({
                "benchmark_version": "ti-benchmark-v1-frozen",
                "case_id": item["case_id"],
                "rank": target.get("rank"),
                "canonical_name": target.get("canonical_name"),
                "gene_symbols": "|".join(target.get("gene_symbols") or []),
                "uniprot_accession": accession,
                "chembl_target": identifiers.get("chembl_target"),
                "evidence_role_suggested": (target.get("scores") or {}).get("explanations", {}).get("evidence_role", "unclassified"),
                "source_evidence_count": (target.get("scores") or {}).get("explanations", {}).get("literature_hits", 0),
                "biological_relevance": (target.get("scores") or {}).get("biological_relevance"),
                "evidence_quality": (target.get("scores") or {}).get("evidence_quality"),
                "qsar_readiness": (target.get("scores") or {}).get("qsar_readiness"),
                "frozen_inclusion_match": "yes" if accession in gold else "no",
                "reviewed_role": "",
                "review_confidence": "",
                "review_notes": "",
            })
    out = root / "docs/target-intelligence/candidate_role_review_set.v1.csv"
    fields = list(rows[0])
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out} with {len(rows)} candidate rows")


if __name__ == "__main__":
    main()
