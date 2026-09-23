"""Generate deterministic report outputs for the frozen benchmark cases."""
from __future__ import annotations

import csv
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from services.target_intelligence import run_target_intelligence_search
from services.target_intelligence import SourceRecord


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cases_path = root / "docs" / "target-intelligence" / "benchmark_cases.v1-frozen.csv"
    output_path = root / "docs" / "target-intelligence" / "benchmark_reports.v1-frozen.jsonl"
    cases = list(csv.DictReader(cases_path.open(encoding="utf-8-sig")))

    def evidence_sources(case: dict[str, str]) -> list[SourceRecord]:
        sources = []
        for pmid in filter(None, (case.get("evidence_pmids") or "").split("|")):
            try:
                payload = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pubmed", "id": pmid, "retmode": "json"}, timeout=30).json()
                item = (payload.get("result") or {}).get(str(pmid)) or {}
                sources.append(SourceRecord(source_type="pubmed", stable_id=f"PMID:{pmid}", title=item.get("title"), url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", publication_year=None, evidence_tier="A"))
            except (requests.RequestException, ValueError, TypeError):
                sources.append(SourceRecord(source_type="pubmed", stable_id=f"PMID:{pmid}", url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", evidence_tier="A"))
        return sources

    def generate(case: dict[str, str]) -> tuple[str, dict]:
        report = run_target_intelligence_search(case["question"], max_targets=10, llm_enhance=False, additional_sources=evidence_sources(case))
        report["benchmark"] = {"version": "ti-benchmark-v1-frozen", "case_id": case["case_id"], "generation_mode": "deterministic"}
        return case["case_id"], report

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(generate, case) for case in cases]
        for future in as_completed(futures):
            case_id, report = future.result()
            results[case_id] = report
            print(f"{case_id}: {len(report.get('targets') or [])} targets", flush=True)
    if set(results) != {case["case_id"] for case in cases}:
        raise RuntimeError("Not all frozen benchmark reports were generated")
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps({"case_id": case["case_id"], "report": results[case["case_id"]]}, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path} ({len(results)} reports)")


if __name__ == "__main__":
    main()
