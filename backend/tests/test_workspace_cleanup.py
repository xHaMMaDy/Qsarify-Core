from pathlib import Path
import uuid

from jobs.run_workspace_cleanup import cleanup_once, purge_account_data


class FakeStore:
    def __init__(self, *, account=None, studies=None, artifacts=None, uploaded=None):
        self.account = account or []
        self.studies = studies or []
        self.artifacts = artifacts or []
        self.uploaded = uploaded or []
        self.calls = []

    def request(self, method, path, *, params=None, payload=None):
        self.calls.append((method, path, params or {}, payload))
        if method == "GET" and path == "ti_account_deletion_requests": return self.account
        if method == "GET" and path == "ti_workspaces": return self.studies
        if method == "GET" and path == "ti_model_artifacts": return self.artifacts
        if method == "GET" and path == "ti_uploaded_models": return self.uploaded
        if method == "DELETE" and path == "ti_workspaces": return [{"id": "deleted"}]
        if method == "PATCH" and path == "ti_account_deletion_requests": return [{"id": "updated"}]
        return []


def test_cleanup_dry_run_reports_due_studies_and_accounts_without_mutation():
    study_id = str(uuid.uuid4())
    account_id = str(uuid.uuid4())
    store = FakeStore(studies=[{"id": study_id, "user_id": str(uuid.uuid4()), "status": "archived"}], account=[{"id": account_id, "user_id": str(uuid.uuid4()), "status": "scheduled"}])
    removed = cleanup_once(dry_run=True, store=store)
    assert removed == ["study:" + study_id, "account:" + account_id]
    assert not any(call[0] in {"DELETE", "PATCH"} for call in store.calls)


def test_account_purge_deletes_records_then_owned_files(tmp_path):
    owner_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    uploaded = tmp_path / f"user_{owner_id}" / "uploaded-models" / f"model_{model_id}" / "model.joblib"
    uploaded.parent.mkdir(parents=True)
    uploaded.write_bytes(b"uploaded")
    trained = tmp_path / "user" / "study" / "model.pkl"
    trained.parent.mkdir(parents=True)
    trained.write_bytes(b"trained")
    store = FakeStore(artifacts=[{"artifact_path": str(trained)}], uploaded=[{"id": model_id, "artifact_format": "joblib"}])

    purge_account_data(store, owner_id, Path(tmp_path))

    assert not uploaded.exists()
    assert not trained.exists()
    deleted_tables = {call[1] for call in store.calls if call[0] == "DELETE"}
    assert "ti_workspaces" in deleted_tables
    assert any(call[0] == "DELETE" and call[1] == "ti_uploaded_models" for call in store.calls)
