import importlib

from services.workspace_collection_jobs import process_collection_job


class FakeStore:
    def __init__(self, *, cancel=False):
        self.cancel = cancel
        self.updates = []
        self.events = []
        self.datasets = []
        self.job = {
            "id": "job-1",
            "workspace_id": "workspace-1",
            "workspace_target_id": "target-1",
            "user_id": "user-1",
            "status": "queued",
            "cancel_requested": False,
            "requested_config": {"activity_type": "IC50", "threshold_nm": 10000},
        }

    def request(self, method, path, *, params=None, payload=None):
        if method == "PATCH" and path == "ti_collection_jobs":
            if payload.get("status") == "validating" and self.job["status"] == "queued":
                self.job.update(payload)
                self.updates.append(payload)
                return [self.job.copy()]
            if payload.get("status") == "validating":
                return []
            self.job.update(payload)
            self.updates.append(payload)
            return [self.job.copy()]
        if method == "GET" and path == "ti_collection_jobs":
            if params and params.get("select") == "status,cancel_requested":
                return [{"status": self.job["status"], "cancel_requested": self.cancel}]
            return [self.job.copy()]
        if method == "GET" and path == "ti_workspace_targets":
            return [{
                "id": "target-1",
                "workspace_id": "workspace-1",
                "user_id": "user-1",
                "report_id": "report-1",
                "target_snapshot": {"identifiers": {"uniprot": "P12345"}},
            }]
        if method == "GET" and path == "ti_training_datasets":
            return []
        if method == "POST" and path == "ti_job_events":
            self.events.append(payload)
            return [{"id": "event-1"}]
        if method == "POST" and path == "ti_training_datasets":
            self.datasets.append(payload)
            return [{"id": "dataset-1", "dataset_version": payload["dataset_version"]}]
        raise AssertionError(f"Unexpected store call: {method} {path} {params} {payload}")


def test_collection_job_persists_hashed_dataset(monkeypatch):
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "get_target", lambda _uniprot: {"target_chembl_id": "CHEMBL-T1"})
    monkeypatch.setattr(app, "get_all_bioactivities", lambda *_args: iter([[{"molecule_chembl_id": "CHEMBL1", "standard_value": 10, "standard_units": "nM"}]]))
    monkeypatch.setattr(app, "get_all_molecules_concurrently", lambda *_args: iter([{"CHEMBL1": {"molecule_structures": {"canonical_smiles": "CCO"}, "molecule_properties": {}}}]))
    monkeypatch.setattr(app, "process_and_merge_data", lambda activities, molecules, threshold: [{"smiles": "CCO", "standard_value": activities[0]["standard_value"], "threshold": threshold}])

    store = FakeStore()
    result = process_collection_job("job-1", store=store)

    assert result["status"] == "completed"
    assert len(store.datasets) == 1
    assert store.datasets[0]["record_count"] == 1
    assert len(store.datasets[0]["content_sha256"]) == 64
    assert store.datasets[0]["source_snapshot"]["report_id"] == "report-1"
    assert any(event["event_type"] == "completed" for event in store.events)


def test_collection_job_stops_at_safe_cancellation_boundary(monkeypatch):
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "get_target", lambda _uniprot: {"target_chembl_id": "CHEMBL-T1"})
    store = FakeStore(cancel=True)

    result = process_collection_job("job-1", store=store)

    assert result["status"] == "cancelled"
    assert store.datasets == []
    assert store.job["status"] == "cancelled"


def test_non_claimed_collection_job_is_not_duplicated():
    store = FakeStore()
    store.job["status"] = "completed"

    result = process_collection_job("job-1", store=store)

    assert result["status"] == "completed"
    assert store.updates == []
    assert store.datasets == []
