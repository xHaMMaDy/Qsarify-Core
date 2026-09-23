import csv
import json

from services.target_intelligence_benchmark import BenchmarkInputError, evaluate_benchmark, load_gold_cases, load_reports


def test_benchmark_evaluator_requires_accepted_source_backed_cases(tmp_path):
    cases = tmp_path / "cases.csv"
    with cases.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "uniprot_accession", "evidence_source_ids", "status"])
        writer.writeheader()
        writer.writerow({"case_id": "CASE-001", "uniprot_accession": "", "evidence_source_ids": "", "status": "proposed"})
    try:
        load_gold_cases(cases)
    except BenchmarkInputError as exc:
        assert "No accepted" in str(exc)
    else:
        raise AssertionError("Expected empty proposed template to be rejected")


def test_benchmark_evaluator_aggregates_supplied_cases(tmp_path):
    cases = [{"case_id": "CASE-001", "gold_targets": ["P12345"], "gold_sources": ["PMID:1"]}]
    reports = {
        "CASE-001": {
            "targets": [{"identifiers": {"uniprot": "P12345"}}],
            "provenance": {"sources": [{"stable_id": "PMID:1"}]},
        }
    }
    result = evaluate_benchmark(cases, reports)
    assert result["cases_evaluated"] == 1
    assert result["metrics"]["recall_at_10"] == 1.0


def test_benchmark_evaluator_normalizes_bare_pubmed_ids():
    cases = [{"case_id": "CASE-001", "gold_targets": ["P12345"], "gold_sources": ["123"]}]
    reports = {"CASE-001": {"targets": [{"identifiers": {"uniprot": "P12345"}}], "provenance": {"sources": [{"stable_id": "PMID:123"}]}}}
    result = evaluate_benchmark(cases, reports)
    assert result["metrics"]["citation_correctness"] == 1.0
