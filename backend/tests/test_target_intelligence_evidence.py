from services.evidence_harness import _validate_extraction


def test_evidence_validation_drops_unknown_citations_and_invalid_accessions():
    result = _validate_extraction(
        {
            "targets": [
                {
                    "name": "candidate",
                    "uniprot_accession": "not-an-accession",
                    "confidence": 1.7,
                    "evidence": [
                        {"source_id": "PMID:known", "claim": "supported", "polarity": "supporting"},
                        {"source_id": "PMID:unknown", "claim": "untrusted", "polarity": "supporting"},
                    ],
                }
            ],
            "warnings": [],
        },
        {"PMID:known"},
    )

    target = result["targets"][0]
    assert target["uniprot_accession"] is None
    assert target["confidence"] == 1.0
    assert [item["source_id"] for item in target["evidence"]] == ["PMID:known"]
    assert any("unknown source ID" in warning for warning in result["warnings"])
    assert any("invalid UniProt" in warning for warning in result["warnings"])
