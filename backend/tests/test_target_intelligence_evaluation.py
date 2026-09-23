from services.target_intelligence_evaluation import (
    evaluate_citations,
    evaluate_identifier_normalization,
    evaluate_target_ranking,
    evaluate_unsupported_claims,
)
from services.evidence_harness import _validate_extraction


def test_target_ranking_metrics_are_deterministic():
    result = evaluate_target_ranking(["P12345", "P67890", "P11111"], ["P67890", "P99999"], k=3)
    assert result["hit_count"] == 1
    assert result["precision_at_k"] == 1 / 3
    assert result["reciprocal_rank"] == 0.5


def test_identifier_normalization_rejects_invalid_and_reports_exact_matches():
    result = evaluate_identifier_normalization(["P12345", "not-an-accession"], ["P12345"])
    assert result["valid_predicted_count"] == 1
    assert result["f1"] == 1.0


def test_citation_and_claim_metrics_keep_missing_audits_explicit():
    assert evaluate_citations(["PMID:1", "DOI:10.1/x"], ["PMID:1"])["correctness"] == 0.5
    result = evaluate_unsupported_claims([{"claim": "uncurated"}])
    assert result["unsupported_rate"] is None


def test_evidence_harness_accepts_common_p_uniprot_accessions():
    result = _validate_extraction(
        {"targets": [{"name": "target", "uniprot_accession": "P10636", "evidence": []}], "warnings": []},
        set(),
    )
    assert result["targets"][0]["uniprot_accession"] == "P10636"
