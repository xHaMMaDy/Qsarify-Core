"""Generate deterministic reports with live literature search disabled."""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.target_intelligence import SourceRecord, run_target_intelligence_search


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cases = {r["case_id"]: r for r in csv.DictReader((root / "docs/target-intelligence/benchmark_cases.v1-frozen.csv").open(encoding="utf-8-sig"))}
    snapshots = {json.loads(line)["case_id"]: json.loads(line) for line in (root / "docs/target-intelligence/frozen_source_snapshots.v1.jsonl").open(encoding="utf-8") if line.strip()}
    def generate(case_id: str):
        sources = [SourceRecord(source_type=s["source_type"], stable_id=s["stable_id"], title=s.get("title"), url=s.get("url"), publication_year=None, evidence_tier="A", abstract_or_excerpt=s.get("abstract_or_excerpt")) for s in snapshots[case_id]["sources"]]
        report = run_target_intelligence_search(cases[case_id]["question"], max_targets=10, source_types=(), additional_sources=sources, llm_enhance=False)
        report["benchmark"] = {"version": "ti-benchmark-v1-frozen", "case_id": case_id, "generation_mode": "frozen_source_deterministic"}
        return case_id, report
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(generate, case_id) for case_id in cases]):
            case_id, report = future.result(); results[case_id] = report; print(case_id, len(report.get("targets") or []), flush=True)
    out = root / "docs/target-intelligence/benchmark_reports.v1-frozen-source.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for case_id in cases:
            handle.write(json.dumps({"case_id": case_id, "report": results[case_id]}, ensure_ascii=False) + "\n")
    print(f"Wrote {out} with {len(results)} reports")


if __name__ == "__main__":
    main()
