"""Strict benchmark aggregation for source-backed Target Intelligence cases."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from services.target_intelligence_evaluation import (
    evaluate_citations,
    evaluate_identifier_normalization,
    evaluate_target_ranking,
)


class BenchmarkInputError(ValueError):
    pass


def _split_ids(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[,;|\s]+", str(value or "")) if part.strip()]


def load_gold_cases(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            status = (row.get("status") or "").strip().lower()
            case_id = (row.get("case_id") or "").strip()
            targets = _split_ids(row.get("uniprot_accession"))
            sources = _split_ids(row.get("evidence_source_ids"))
            if status not in {"accepted", "reviewed"}:
                continue
            if not case_id or not targets or not sources:
                raise BenchmarkInputError(f"Accepted case at line {line_number} requires case_id, uniprot_accession, and evidence_source_ids")
            rows.append({"case_id": case_id, "gold_targets": targets, "gold_sources": sources, "expected_qsar_readiness": row.get("expected_qsar_readiness")})
    if not rows:
        raise BenchmarkInputError("No accepted/reviewed source-backed benchmark cases were found; refusing to produce metrics")
    return rows


def load_reports(path: str | Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkInputError(f"Invalid report JSON on line {line_number}") from exc
            case_id = item.get("case_id")
            report = item.get("report", item)
            if not isinstance(case_id, str) or not case_id or not isinstance(report, dict):
                raise BenchmarkInputError(f"Report line {line_number} requires case_id and report object")
            reports[case_id] = report
    return reports


def evaluate_benchmark(cases: list[dict[str, Any]], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [case["case_id"] for case in cases if case["case_id"] not in reports]
    if missing:
        raise BenchmarkInputError(f"Missing report outputs for cases: {', '.join(missing)}")
    per_case: list[dict[str, Any]] = []
    for case in cases:
        report = reports[case["case_id"]]
        targets = report.get("targets") or []
        predicted_ids = [((target.get("identifiers") or {}).get("uniprot")) for target in targets if isinstance(target, dict)]
        selective_ids = [((target.get("identifiers") or {}).get("uniprot")) for target in targets if isinstance(target, dict) and (target.get("normalization") or {}).get("status") == "canonical_candidate"]
        predicted_sources = [source.get("stable_id") for source in (report.get("provenance") or {}).get("sources", []) if isinstance(source, dict)]
        ranking = {
            "at_1": evaluate_target_ranking(predicted_ids, case["gold_targets"], k=1),
            "at_5": evaluate_target_ranking(predicted_ids, case["gold_targets"], k=5),
            "at_10": evaluate_target_ranking(predicted_ids, case["gold_targets"], k=10),
        }
        normalization = evaluate_identifier_normalization(predicted_ids, case["gold_targets"])
        selective_normalization = evaluate_identifier_normalization(selective_ids, case["gold_targets"])
        citations = evaluate_citations(predicted_sources, case["gold_sources"])
        per_case.append({"case_id": case["case_id"], "ranking": ranking, "normalization": normalization, "selective_normalization": selective_normalization, "selective_coverage": len(selective_ids) / len(predicted_ids) if predicted_ids else 0.0, "citations": citations})

    def avg(path: tuple[str, str, str]) -> float:
        values = [item[path[0]][path[1]][path[2]] for item in per_case]
        return mean(float(value) for value in values if value is not None)

    def avg_flat(path: tuple[str, str]) -> float:
        return mean(float(item[path[0]][path[1]]) for item in per_case)

    return {
        "cases_evaluated": len(per_case),
        "metrics": {
            "precision_at_1": avg(("ranking", "at_1", "precision_at_k")),
            "precision_at_5": avg(("ranking", "at_5", "precision_at_k")),
            "recall_at_10": avg(("ranking", "at_10", "recall_at_k")),
            "mean_reciprocal_rank": avg(("ranking", "at_10", "reciprocal_rank")),
            "ndcg_at_10": avg(("ranking", "at_10", "ndcg_at_k")),
            "normalization_f1": avg_flat(("normalization", "f1")),
            "selective_normalization_f1": avg_flat(("selective_normalization", "f1")),
            "selective_normalization_coverage": mean(float(item["selective_coverage"]) for item in per_case),
            "citation_correctness": avg_flat(("citations", "correctness")),
        },
        "per_case": per_case,
        "disclaimer": "Metrics describe only the supplied source-backed benchmark cases and saved report outputs; they are not a scientific claim until reviewed.",
    }
