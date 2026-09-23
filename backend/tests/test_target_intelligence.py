from services.target_intelligence import SourceRecord, TargetCandidate, TargetIntelligenceError, _modality_guard, _safe_search_terms, _score_candidate, enrich_target_with_chembl, normalize_weights, run_target_intelligence_search
import services.target_intelligence as target_intelligence
from services import target_intelligence_config
from services.reference_sources import ReferenceInputError, resolve_reference_input
import app as app_module


def test_normalize_weights_accepts_configured_weights():
    weights = normalize_weights({
        "biological_relevance": 0.5,
        "evidence_quality": 0.25,
        "qsar_readiness": 0.25,
    })
    assert weights["biological_relevance"] == 0.5
    assert sum(weights.values()) == 1.0


def test_normalize_weights_rejects_non_unit_sum():
    try:
        normalize_weights({
            "biological_relevance": 0.5,
            "evidence_quality": 0.5,
            "qsar_readiness": 0.5,
        })
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("Expected non-unit ranking weights to be rejected")


def test_search_input_validation_happens_before_network_calls():
    try:
        run_target_intelligence_search("", max_targets=10)
    except ValueError as exc:
        assert "at least 3" in str(exc)
    else:
        raise AssertionError("Expected short question to be rejected")


def test_search_rejects_target_limit_outside_public_contract():
    try:
        run_target_intelligence_search("kinase inhibitors", max_targets=11)
    except ValueError as exc:
        assert "between 1 and 10" in str(exc)
    else:
        raise AssertionError("Expected target limit to be rejected")


def test_safe_search_terms_prefers_disease_phrase_over_later_qsar_clause():
    terms = _safe_search_terms(
        "Which targets are relevant to Parkinson's disease and have reliable ChEMBL data for QSAR modeling?"
    )
    assert "Parkinson" in terms
    assert "disease" in terms
    assert "QSAR" not in terms


def test_safe_search_terms_extracts_breast_cancer_phrase():
    terms = _safe_search_terms(
        "Which human protein targets are associated with breast cancer but currently lack sufficient chemical data for QSAR?"
    )
    assert "breast" in terms
    assert "cancer" in terms
    assert "QSAR" not in terms


def test_safe_search_terms_preserves_explicit_comparison_targets():
    terms = _safe_search_terms(
        "Compare APP, MAPT, BACE1, and ACHE for Alzheimer's disease by evidence quality and QSAR readiness."
    )
    assert all(token in terms for token in ("APP", "MAPT", "BACE1", "ACHE"))
    assert "QSAR" not in terms


def test_openalex_receives_sanitized_search_terms(monkeypatch):
    captured = {}

    def fake_request(url, *, params=None, **_kwargs):
        captured["url"] = url
        captured["params"] = params
        return {"results": []}

    monkeypatch.setattr(target_intelligence, "_request_json", fake_request)
    target_intelligence.search_openalex(
        "Which targets are relevant to Parkinson's disease and have reliable ChEMBL data for QSAR modeling?"
    )

    assert captured["url"] == "https://api.openalex.org/works"
    assert captured["params"]["search"] == "Parkinson disease"


def test_transient_chembl_failure_does_not_claim_qsar_unready(monkeypatch):
    def unavailable(_accession):
        raise TargetIntelligenceError("source timeout")

    monkeypatch.setattr("services.target_intelligence._cached_chembl_enrichment", unavailable)
    candidate = TargetCandidate("Example target", "P12345", True)
    enrich_target_with_chembl(candidate)

    scores = _score_candidate(candidate, {"biological_relevance": 0.4, "evidence_quality": 0.3, "qsar_readiness": 0.3})
    assert scores["qsar_readiness"] is None
    assert "QSAR readiness unavailable" in scores["labels"]
    assert "Biologically relevant, QSAR-unready" not in scores["labels"]
    assert scores["explanations"]["chembl_status"] == "temporarily_unavailable"


def test_missing_chembl_mapping_remains_explicitly_qsar_unready(monkeypatch):
    monkeypatch.setattr("services.target_intelligence._cached_chembl_enrichment", lambda _accession: {"status": "not_found"})
    candidate = TargetCandidate("Unmapped target", "P12345", True)
    enrich_target_with_chembl(candidate)

    scores = _score_candidate(candidate, {"biological_relevance": 0.4, "evidence_quality": 0.3, "qsar_readiness": 0.3})
    assert scores["qsar_readiness"] == 10.0
    assert "Biologically relevant, QSAR-unready" in scores["labels"]
    assert scores["explanations"]["chembl_status"] == "not_found"


def test_modality_guard_blocks_biologic_accession():
    candidate = TargetCandidate("Interleukin-5", "P05113", True, chembl_status="ok", chembl_activity_count=3)
    modality, reasons = _modality_guard(candidate, [])
    assert modality == "biologic_or_antibody_only"
    assert reasons


def test_modality_guard_requires_chembl_activity():
    candidate = TargetCandidate("Example kinase", "P12345", True, chembl_status="not_found", chembl_activity_count=None)
    modality, reasons = _modality_guard(candidate, [])
    assert modality == "modality_unknown"
    assert reasons


def test_modality_guard_detects_antibody_evidence():
    candidate = TargetCandidate("Example target", "P12345", True, chembl_status="ok", chembl_activity_count=100, supporting_source_ids=["PMID:1"])
    sources = [SourceRecord("pubmed", "PMID:1", abstract_or_excerpt="A monoclonal antibody against the target was tested.")]
    modality, _ = _modality_guard(candidate, sources)
    assert modality == "biologic_or_antibody_only"


def test_runtime_model_allowlist_and_token_ceiling_are_server_side(monkeypatch):
    monkeypatch.setattr(
        target_intelligence_config,
        "_cache",
        {
            "checked_at": 9999999999.0,
            "settings": {"max_response_tokens": 512, "allowed_models": ["allowed-model"]},
        },
    )
    assert target_intelligence_config.model_is_allowed("allowed-model")
    assert not target_intelligence_config.model_is_allowed("unlisted-model")
    assert target_intelligence_config.max_response_tokens() == 512


def test_reference_inputs_preserve_provenance_and_reject_private_urls(monkeypatch):
    def fake_request(url, **kwargs):
        if "crossref" in url:
            return {"message": {"title": ["A DOI record"], "DOI": "10.1000/example", "URL": "https://doi.org/10.1000/example", "published": {"date-parts": [[2024]]}}}
        return {"result": {"12345678": {"title": "A PubMed record", "fulljournalname": "Journal"}}}

    monkeypatch.setattr("services.reference_sources._request_json", fake_request)
    doi = resolve_reference_input("doi:10.1000/example")
    pmid = resolve_reference_input("PMID:12345678")
    pasted = resolve_reference_input("This is bounded user-provided evidence.")
    assert doi.source_type == "crossref" and doi.doi == "10.1000/example"
    assert pmid.stable_id == "PMID:12345678"
    assert pasted.source_type == "user_text" and pasted.metadata["input_kind"] == "pasted_text"
    try:
        resolve_reference_input("http://127.0.0.1:8080/private")
    except ReferenceInputError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("Expected non-HTTPS URL to be rejected")


def test_search_endpoint_rejects_invalid_payload_without_network_call(monkeypatch):
    monkeypatch.setattr(app_module, "SUPABASE_URL", "")
    monkeypatch.setattr(app_module, "SUPABASE_JWT_SECRET", "")
    monkeypatch.setattr(app_module, "FLASK_ENV", "development")
    client = app_module.app.test_client()
    response = client.post("/api/target-intelligence/search", json={"question": "x"})
    assert response.status_code == 400
    assert "at least 3" in response.get_json()["error"]
    assert response.get_json()["request_id"] == response.headers["X-Request-ID"]


def test_usage_event_recording_excludes_prompt_and_ip(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(app_module, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    request_obj = type("Request", (), {"remote_addr": "192.0.2.10", "user": None})()

    app_module._record_target_intelligence_usage(
        "chat_request",
        request_obj,
        {"model": "test-model", "usage": {"prompt_tokens": 4, "completion_tokens": 6}, "prompt": "do not persist"},
    )

    payload = captured["json"]
    assert payload["event_type"] == "chat_request"
    assert payload["input_tokens"] == 4
    assert payload["output_tokens"] == 6
    assert "prompt" not in payload
    assert "remote_addr" not in payload
    assert payload["anonymous_run_hash"]


def test_analysis_quota_uses_durable_event_count(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"id": "one"}, {"id": "two"}]

    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(app_module, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(app_module, "_target_intelligence_daily_limit", lambda authenticated: 2)
    monkeypatch.setattr(app_module, "_consume_target_intelligence_quota", lambda *_args: None)
    monkeypatch.setattr(app_module.requests, "get", fake_get)
    request_obj = type("Request", (), {"remote_addr": "192.0.2.10", "user": None})()

    with app_module.app.app_context():
        response = app_module._target_intelligence_quota_response("analysis_started", request_obj)

    assert response[1] == 429
    assert response[0].get_json()["code"] == "quota_exceeded"
    assert calls[0][1]["params"]["event_type"] == "eq.analysis_started"


def test_atomic_quota_rpc_consumes_slot_without_duplicate_analysis_event(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return True

    captured = {}

    def fake_post(url, **kwargs):
        captured.update({"url": url, "kwargs": kwargs})
        return FakeResponse()

    monkeypatch.setattr(app_module, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(app_module, "_target_intelligence_daily_limit", lambda authenticated: 10)
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    request_obj = type("Request", (), {"remote_addr": "192.0.2.10", "user": None, "environ": {}})()

    result = app_module._target_intelligence_quota_response("analysis_started", request_obj)

    assert result is None
    assert request_obj.environ["_ti_quota_consumed"] is True
    assert captured["url"].endswith("/rest/v1/rpc/consume_ti_daily_quota")
    assert captured["kwargs"]["json"]["p_daily_limit"] == 10


def test_chat_quota_reads_separate_chat_setting(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"public_daily_chat_requests": 30, "authenticated_daily_chat_requests": 100}]

    monkeypatch.setattr(app_module, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(app_module.requests, "get", lambda *_args, **_kwargs: FakeResponse())

    assert app_module._target_intelligence_chat_daily_limit(False) == 30
    assert app_module._target_intelligence_chat_daily_limit(True) == 100
