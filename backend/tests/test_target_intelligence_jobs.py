from services.literature_uploads import extract_literature_text
from services.target_intelligence_jobs import SupabaseWorkerStore, load_upload_sources
import services.target_intelligence_jobs as jobs
from services.target_intelligence import SourceRecord


def test_extract_plain_text_is_bounded_and_decoded():
    text = extract_literature_text("\ufeffTarget evidence".encode("utf-8"), "text/plain")
    assert text == "Target evidence"


def test_extract_rejects_unsupported_content_type():
    try:
        extract_literature_text(b"content", "application/octet-stream")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Expected unsupported content type to be rejected")


def test_worker_store_requires_server_side_credentials(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    try:
        SupabaseWorkerStore()
    except RuntimeError as exc:
        assert "SUPABASE_SERVICE_ROLE_KEY" in str(exc)
    else:
        raise AssertionError("Expected worker credential configuration to be required")


def test_upload_extraction_failures_are_explicitly_reported_and_excluded():
    class FakeStore:
        def get_uploads(self, _upload_ids, _user_id):
            return [{"id": "upload-1", "original_filename": "bad.bin", "content_type": "application/octet-stream"}]

        def download_upload(self, _storage_path):
            return b"not supported"

    warnings = []
    sources = load_upload_sources(FakeStore(), ["upload-1"], "user-1", warnings)
    assert sources == []
    assert warnings[0]["source_type"] == "upload"
    assert "bad.bin" in warnings[0]["error"]


def test_worker_passes_reference_inputs_into_report_context(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.updates = []
            self.inserted = None

        def get_run(self, _run_id):
            return {
                "id": "run-1",
                "user_id": "user-1",
                "question": "Alzheimer disease target prioritization",
                "reference_inputs": ["PMID:12345678"],
                "upload_ids": [],
                "ranking_weights": None,
                "requested_sources": ["uniprot"],
                "status": "created",
                "attempt_count": 0,
            }

        def update_run(self, _run_id, payload):
            self.updates.append(payload)

        def claim_run(self, _run_id):
            return {**self.get_run(_run_id), "status": "validating"}

        def get_uploads(self, _upload_ids, _user_id):
            return []

        def insert_report(self, payload):
            self.inserted = payload
            return {"id": "report-1"}

    store = FakeStore()
    reference = SourceRecord("pubmed", "PMID:12345678", title="Referenced record", evidence_tier="E")
    monkeypatch.setattr(jobs, "resolve_reference_inputs", lambda _values: ([reference], []))
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return {"report_version": "ti-retrieval-v2", "provenance": {"source_errors": []}, "targets": [], "limitations": []}

    monkeypatch.setattr(jobs, "run_target_intelligence_search", fake_search)
    result = jobs.process_run("run-1", store=store)

    assert result["status"] == "completed"
    assert captured["additional_sources"] == [reference]
    assert store.inserted["report"]["provenance"]["source_errors"] == []


def test_worker_store_claim_is_atomic_and_filters_created_state():
    store = SupabaseWorkerStore.__new__(SupabaseWorkerStore)
    captured = {}

    def fake_request(method, path, *, params=None, payload=None):
        captured.update({"method": method, "path": path, "params": params, "payload": payload})
        return [{"id": "run-1", "status": "validating"}]

    store.request = fake_request
    claimed = store.claim_run("run-1")
    assert claimed["status"] == "validating"
    assert captured["params"]["status"] == "eq.created"
    assert captured["payload"]["progress_stage"] == "validating"


def test_training_dataset_lookup_uses_supabase_rest_and_owner_scope():
    store = SupabaseWorkerStore.__new__(SupabaseWorkerStore)
    captured = {}
    expected = [{"id": "00000000-0000-0000-0000-000000000020", "status": "ready_for_training"}]

    def fake_request(method, path, *, params=None, payload=None):
        captured.update({"method": method, "path": path, "params": params, "payload": payload})
        return expected

    store.request = fake_request
    rows = store.get_training_datasets_for_run(
        ["00000000-0000-0000-0000-000000000020"],
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000099",
    )

    assert rows == expected
    assert captured["method"] == "GET"
    assert captured["path"] == "ti_training_datasets"
    assert captured["params"]["workspace_id"] == "eq.00000000-0000-0000-0000-000000000001"
    assert captured["params"]["user_id"] == "eq.00000000-0000-0000-0000-000000000099"
    assert captured["params"]["select"] == "id,workspace_target_id,status,curated_records,records"


def test_training_dataset_lookup_rejects_malformed_ids_before_request():
    store = SupabaseWorkerStore.__new__(SupabaseWorkerStore)
    store.request = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected request"))

    assert store.get_training_datasets_for_run(
        ["not-a-uuid"],
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000099",
    ) == []


def test_pending_worker_skips_run_claimed_by_another_worker():
    class FakeStore:
        def request(self, *_args, **_kwargs):
            return [{"id": "run-1", "status": "created"}]

        def claim_run(self, _run_id):
            return None

    result = jobs.process_pending_runs(limit=1, store=FakeStore())
    assert result == [{"run_id": "run-1", "status": "skipped", "reason": "claimed_by_other_worker"}]
