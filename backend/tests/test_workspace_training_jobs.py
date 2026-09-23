from pathlib import Path

from services.workspace_training_jobs import TrainingCancelled, _check_cancel, process_training_run


class FakeTrainingStore:
    def __init__(self):
        self.run = {
            "id": "00000000-0000-0000-0000-000000000010",
            "workspace_id": "00000000-0000-0000-0000-000000000001",
            "user_id": "00000000-0000-0000-0000-000000000099",
            "dataset_ids": ["00000000-0000-0000-0000-000000000020"],
            "task_type": "classification",
            "architecture": "separate_models",
            "model_names": ["random_forest"],
            "status": "queued",
            "cancel_requested": False,
            "attempt_count": 1,
            "expires_at": None,
        }
        self.artifacts = []

    def get_training_datasets_for_run(self, dataset_ids, workspace_id, user_id):
        return self.request(
            "GET",
            "ti_training_datasets",
            params={"id": f"in.({','.join(dataset_ids)})"},
        )

    def request(self, method, path, *, params=None, payload=None):
        if path == "ti_model_artifacts" and method == "POST":
            self.artifacts = list(payload or [])
            return [{"id": f"artifact-{index}"} for index, _ in enumerate(self.artifacts, start=1)]
        if path == "ti_training_runs" and method == "GET":
            return [self.run]
        if path == "ti_training_runs" and method == "PATCH":
            if params and params.get("status") == "eq.queued" and self.run["status"] != "queued":
                return []
            self.run.update(payload or {})
            return [self.run]
        if path == "ti_training_datasets" and method == "GET":
            return [{
                "id": self.run["dataset_ids"][0],
                "workspace_target_id": "00000000-0000-0000-0000-000000000030",
                "status": "ready_for_training",
                "curated_records": [
                    {"smiles": smiles, "bioactivity_class": "Active" if index % 2 else "Inactive", "scaffold": f"scaffold-{index}"}
                    for index, smiles in enumerate(["CCO", "CCN", "CCC", "CCCl", "CCBr", "C1CCCCC1", "c1ccccc1", "COC", "CNC", "CC(=O)O"])
                ],
                "records": [],
            }]
        if path == "ti_workspace_targets" and method == "GET":
            return [{"id": "00000000-0000-0000-0000-000000000030", "target_key": "P12345", "target_snapshot": {"identifiers": {"uniprot": "P12345"}}}]
        if path == "ti_model_artifacts" and method == "POST":
            self.artifacts = [{"id": f"artifact-{index}"} for index, _ in enumerate(payload or [], start=1)]
            return self.artifacts
        raise AssertionError(f"Unexpected {method} {path} request")


class BothTrainingStore(FakeTrainingStore):
    def __init__(self):
        super().__init__()
        parent_id = self.run["id"]
        self.run.update({"architecture": "both", "dataset_ids": ["00000000-0000-0000-0000-000000000020", "00000000-0000-0000-0000-000000000021"], "execution_scope": "parent"})
        self.children = [
            {**self.run, "id": "00000000-0000-0000-0000-000000000011", "parent_run_id": parent_id, "architecture": "separate_models", "execution_scope": "separate_child"},
            {**self.run, "id": "00000000-0000-0000-0000-000000000012", "parent_run_id": parent_id, "architecture": "pooled_multitarget", "execution_scope": "pooled_child"},
        ]

    def request(self, method, path, *, params=None, payload=None):
        if path == "ti_training_runs" and method == "GET":
            if params and "parent_run_id" in params:
                return self.children
            if params and params.get("id"):
                requested = str(params["id"]).removeprefix("eq.")
                return [row for row in [self.run, *self.children] if row["id"] == requested]
            return [self.run, *self.children]
        if path == "ti_training_runs" and method == "PATCH":
            requested = str((params or {}).get("id", "")).removeprefix("eq.")
            rows = [row for row in [self.run, *self.children] if row["id"] == requested]
            if not rows or ((params or {}).get("status") == "eq.queued" and rows[0]["status"] != "queued"):
                return []
            rows[0].update(payload or {})
            return rows
        if path == "ti_training_datasets" and method == "GET":
            base = [
                {"id": self.run["dataset_ids"][0], "workspace_target_id": "00000000-0000-0000-0000-000000000030", "status": "ready_for_training"},
                {"id": self.run["dataset_ids"][1], "workspace_target_id": "00000000-0000-0000-0000-000000000031", "status": "ready_for_training"},
            ]
            rows = []
            for item in base:
                records = [{"smiles": smiles, "bioactivity_class": "Active" if index % 2 else "Inactive", "scaffold": f"{item['workspace_target_id']}-{index}"} for index, smiles in enumerate(["CCO", "CCN", "CCC", "CCCl", "CCBr", "C1CCCCC1", "c1ccccc1", "COC", "CNC", "CC(=O)O"])]
                rows.append({**item, "curated_records": records, "records": []})
            return rows
        if path == "ti_workspace_targets" and method == "GET":
            return [
                {"id": "00000000-0000-0000-0000-000000000030", "target_key": "P12345", "target_snapshot": {"identifiers": {"uniprot": "P12345"}}},
                {"id": "00000000-0000-0000-0000-000000000031", "target_key": "P54321", "target_snapshot": {"identifiers": {"uniprot": "P54321"}}},
            ]
        return super().request(method, path, params=params, payload=payload)


def test_training_job_persists_progress_and_artifact_ids(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(tmp_path))
    store = FakeTrainingStore()
    result = process_training_run(store.run["id"], store=store)
    assert result["status"] == "ready_for_review"
    assert store.run["progress_percent"] == 100
    assert store.run["artifact_ids"]
    assert store.artifacts


def test_training_job_rejects_pooled_run_with_one_target(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(tmp_path))
    store = FakeTrainingStore()
    store.run["architecture"] = "pooled_multitarget"
    try:
        process_training_run(store.run["id"], store=store)
    except ValueError as error:
        assert "at least two selected target identities" in str(error)
    else:
        raise AssertionError("Expected pooled single-target validation to fail")
    assert store.run["status"] == "failed"


def test_training_job_honors_cancellation_request():
    store = FakeTrainingStore()
    store.run["cancel_requested"] = True
    try:
        _check_cancel(store, store.run["id"])
    except TrainingCancelled:
        assert store.run["status"] == "cancelled"
    else:
        raise AssertionError("Expected cancellation request to stop the worker")


def test_both_training_persists_parent_and_child_scope_results(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(tmp_path))
    store = BothTrainingStore()
    result = process_training_run(store.run["id"], store=store)
    assert result["status"] == "ready_for_review"
    assert all(child["status"] == "completed" for child in store.children)
    assert {row["training_run_id"] for row in store.artifacts} == {child["id"] for child in store.children}
