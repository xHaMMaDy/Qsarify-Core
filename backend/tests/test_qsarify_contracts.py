"""Deterministic smoke tests for the QSARify scientific/API contracts.

These tests intentionally exercise the same helpers used by the Flask service.
They are lightweight release-gate tests, not a replacement for the manuscript
benchmark suite.
"""

import math
import hashlib
import json
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as qsarify_app  # noqa: E402

BUILTIN_MODEL_PATH = Path(qsarify_app.__file__).resolve().parent / "Model" / "final_tuned_model.pkl"
BUILTIN_MODEL_METADATA_PATH = BUILTIN_MODEL_PATH.with_suffix(".metadata.json")


def _require_bundled_model_fixture():
    if not BUILTIN_MODEL_PATH.is_file() or not BUILTIN_MODEL_METADATA_PATH.is_file():
        pytest.skip("This contract test requires the separately licensed bundled model artifact, which is excluded from the public source archive.")
from services.applicability_domain import (
    AD_STATUS_BORDERLINE,
    AD_STATUS_IN_DOMAIN,
    AD_STATUS_OUT_OF_DOMAIN,
    AD_STATUS_UNAVAILABLE,
    assess_molecule_applicability_domain,
    pack_fingerprint_matrix,
    similarity_status,
)


def test_morgan_fingerprint_has_requested_width():
    molecule = Chem.MolFromSmiles("CCO")

    fingerprint = qsarify_app.get_fingerprints(molecule, "Morgan", radius=2, nBits=128)

    assert fingerprint is not None
    assert len(fingerprint) == 128


def test_supported_fingerprint_types_have_requested_width():
    molecule = Chem.MolFromSmiles("c1ccccc1O")

    for fingerprint_type in ("Morgan", "RDKit", "Topological"):
        fingerprint = qsarify_app.get_fingerprints(molecule, fingerprint_type, radius=2, nBits=256)
        assert fingerprint is not None, fingerprint_type
        assert len(fingerprint) == 256


def test_default_physicochemical_descriptor_contract():
    molecule = Chem.MolFromSmiles("CCO")
    names = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"]

    values = qsarify_app.calculate_descriptors(molecule, names)

    assert len(values) == len(names)
    assert all(math.isfinite(float(value)) for value in values)


def test_imputation_is_fit_from_training_partition_only():
    X = pd.DataFrame({"feature": np.arange(1.0, 21.0)})
    y = pd.Series([0, 1] * 10, name="bioactivity_class")
    _, X_test_indices, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=qsarify_app.SEED,
        stratify=y,
    )
    missing_index = X_test_indices.index[0]
    X.loc[missing_index, "feature"] = np.nan
    params = {"test_size": 0.2, "use_resampling": False, "physchem": {"missing_strategy": "mean"}}

    X_train, X_test, _, _ = qsarify_app._split_and_prepare_training_data(X, y, params)
    training_mean = X_train["feature"].mean()

    assert X_test.loc[missing_index, "feature"] == pytest.approx(training_mean)


def test_tuning_pipeline_contains_preprocessing_inside_cross_validation():
    y_train = pd.Series([0, 1] * 6, name="bioactivity_class")
    params = {"use_resampling": False, "physchem": {"missing_strategy": "mean"}}
    model = qsarify_app.get_models()["LogisticRegression"]
    estimator, grid = qsarify_app._build_tuning_pipeline(
        model,
        {"C": [0.1, 1.0]},
        params,
        y_train,
    )

    assert list(estimator.named_steps) == ["imputer", "model"]
    assert set(grid) == {"model__C"}


def test_tuning_pipeline_keeps_resampling_inside_cross_validation():
    y_train = pd.Series([0, 1] * 6, name="bioactivity_class")
    params = {"use_resampling": True, "physchem": {"missing_strategy": "mean"}}
    model = qsarify_app.get_models()["LogisticRegression"]
    estimator, _ = qsarify_app._build_tuning_pipeline(model, {"C": [1.0]}, params, y_train)

    assert list(estimator.named_steps) == ["imputer", "resampler", "model"]


def test_training_pipeline_persists_selected_preprocessing_boundary():
    y_train = pd.Series([0, 1] * 8, name="bioactivity_class")
    params = {"use_resampling": True, "physchem": {"missing_strategy": "mean"}}
    model = qsarify_app.get_models()["LogisticRegression"]

    estimator = qsarify_app._build_preprocessing_pipeline(model, params, y_train)

    assert list(estimator.named_steps) == ["imputer", "resampler", "model"]


def test_raw_training_split_keeps_missing_values_for_pipeline_fit():
    X = pd.DataFrame({"feature": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1], name="bioactivity_class")
    params = {"test_size": 0.33, "use_resampling": False, "physchem": {"missing_strategy": "mean"}}

    X_train, X_test, _, _ = qsarify_app._split_raw_tuning_data(X, y, params)

    assert X_train["feature"].isna().any() or X_test["feature"].isna().any()

    estimator = qsarify_app._build_preprocessing_pipeline(
        qsarify_app.get_models()["LogisticRegression"], params, y_train=y.loc[X_train.index]
    )
    estimator.fit(X_train, y.loc[X_train.index])
    assert np.isfinite(estimator.predict_proba(X_test)[:, 1]).all()


def test_bounded_ttl_store_evicts_expired_and_old_entries(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(qsarify_app.time, "monotonic", lambda: now[0])
    store = qsarify_app.BoundedTTLStore(max_entries=2, ttl_seconds=10)

    store["a"] = {"status": "pending"}
    store["b"] = {"status": "pending"}
    store["c"] = {"status": "pending"}

    assert len(store) == 2
    assert store.get("a") is None
    assert store.get("b")["status"] == "pending"

    now[0] = 11.0
    assert store.get("b") is None
    assert len(store) == 0


def test_model_registry_contains_paper_model_families():
    models = qsarify_app.get_models()
    expected = {
        "LogisticRegression",
        "KNeighborsClassifier",
        "SVC",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "AdaBoostClassifier",
        "MLPClassifier",
        "GaussianNB",
    }

    assert expected.issubset(models)


def test_prediction_endpoint_requires_authentication(monkeypatch):
    # Do not depend on a developer's local .env file. A configured auth
    # provider is enough to exercise the missing-header boundary; no token
    # verification request is made for this case.
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "https://auth.example.invalid")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    client = qsarify_app.app.test_client()

    response = client.post("/predict", json={})

    assert response.status_code == 401
    assert "error" in response.get_json()


def test_debug_test_endpoint_is_not_public(monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "https://auth.example.invalid")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    client = qsarify_app.app.test_client()

    response = client.get("/api/test")

    assert response.status_code == 401


def test_request_logging_does_not_persist_json_values(caplog):
    with qsarify_app.app.test_request_context(
        "/api/predict",
        method="POST",
        json={"password": "do-not-log", "smiles": "CCO", "target_protein": "AChE"},
    ):
        with caplog.at_level("DEBUG", logger=qsarify_app.app_logger.name):
            qsarify_app.log_request_info()

    assert "do-not-log" not in caplog.text
    assert "Request JSON fields" in caplog.text


def test_process_data_route_does_not_write_raw_request_debug_log(tmp_path, monkeypatch):
    monkeypatch.setattr(qsarify_app, "LOG_DIR", str(tmp_path))
    with qsarify_app.app.test_request_context(
        "/api/training/process-data",
        method="POST",
        json={"smiles": "CCO", "password": "must-not-be-written"},
    ):
        response = qsarify_app.api_process_data()

    body, status = response
    assert status == 400
    assert "associated with this session" in body.get_json()["error"]
    assert not (tmp_path / "debug.log").exists()


def test_processed_data_requires_session_owner(monkeypatch):
    data_id = "owned-data-for-test"
    owner_id = "owner-a-for-test"
    qsarify_app.processed_data_store[data_id] = {
        "owner_id": "different-owner",
        "processed_data": {"X": "{}", "y": "{}"},
    }
    with qsarify_app.app.test_request_context("/api/training/train", method="POST"):
        qsarify_app.session["data_id"] = data_id
        qsarify_app.session[qsarify_app.TASK_OWNER_SESSION_KEY] = owner_id
        assert qsarify_app._get_owned_processed_data(data_id) is None


def test_training_rejects_data_id_from_another_session(monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    client = qsarify_app.app.test_client()
    with client.session_transaction() as session_data:
        session_data["data_id"] = "session-owned-data"

    response = client.post("/train", json={"data_id": "another-session-data"})

    assert response.status_code == 403
    assert "not owned by this session" in response.get_json()["error"]


def test_task_status_is_owner_bound_and_hides_owner_token(monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    client = qsarify_app.app.test_client()
    owner_id = "task-owner-a-" + "x" * 32
    with client.session_transaction() as session_data:
        session_data[qsarify_app.TASK_OWNER_SESSION_KEY] = owner_id
    task_id = "task-owner-bound-test"
    qsarify_app.tasks[task_id] = {
        "owner_id": owner_id,
        "status": "complete",
        "result": {"ok": True},
    }

    response = client.get(f"/task-status/{task_id}")
    assert response.status_code == 200
    assert response.get_json() == {"status": "complete", "result": {"ok": True}}

    other_client = qsarify_app.app.test_client()
    assert other_client.get(f"/task-status/{task_id}").status_code == 404


def test_generated_model_artifact_directories_are_owner_scoped(tmp_path, monkeypatch):
    monkeypatch.setitem(qsarify_app.app.config, "MODEL_FOLDER", str(tmp_path))
    first = qsarify_app._get_owned_model_folder("a" * 32)
    second = qsarify_app._get_owned_model_folder("b" * 32)

    assert first != second
    assert Path(first).parent == Path(second).parent
    assert Path(first).is_dir()
    assert Path(second).is_dir()


def test_owner_namespace_changes_with_authenticated_subject():
    with qsarify_app.app.test_request_context("/api/training/models-and-params"):
        qsarify_app.request.user = {"sub": "supabase-user-a"}
        first = qsarify_app._get_task_owner_id()
        qsarify_app.request.user = {"sub": "supabase-user-b"}
        second = qsarify_app._get_task_owner_id()

    assert first != second
    assert len(first) >= 32
    assert len(second) >= 32


def test_training_reset_preserves_owner_namespace(monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    owner_id = "reset-owner-" + "z" * 28
    client = qsarify_app.app.test_client()
    with client.session_transaction() as session_data:
        session_data[qsarify_app.TASK_OWNER_SESSION_KEY] = owner_id

    response = client.post("/reset")

    assert response.status_code == 200
    with client.session_transaction() as session_data:
        assert session_data[qsarify_app.TASK_OWNER_SESSION_KEY] == owner_id


def test_model_download_reads_only_the_current_session_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    monkeypatch.setitem(qsarify_app.app.config, "MODEL_FOLDER", str(tmp_path))
    owner_id = "download-owner-" + "x" * 24
    other_owner_id = "download-other-" + "y" * 23
    filename = "model_RandomForestClassifier.pkl"
    Path(qsarify_app._get_owned_model_folder(other_owner_id), filename).write_bytes(b"other-user-model")

    client = qsarify_app.app.test_client()
    with client.session_transaction() as session_data:
        session_data[qsarify_app.TASK_OWNER_SESSION_KEY] = owner_id
    response = client.get("/download-model?model_name=RandomForestClassifier")
    assert response.status_code == 404

    Path(qsarify_app._get_owned_model_folder(owner_id), filename).write_bytes(b"current-user-model")
    response = client.get("/download-model?model_name=RandomForestClassifier")
    assert response.status_code == 200
    assert response.data == b"current-user-model"


def test_model_artifact_metadata_records_digest_and_feature_schema(tmp_path):
    model_filename = "model_RandomForestClassifier.pkl"
    model_path = tmp_path / model_filename
    model_path.write_bytes(b"model-payload")

    metadata_filename, metadata = qsarify_app._write_model_artifact_metadata(
        str(tmp_path),
        model_filename,
        model_name="RandomForestClassifier",
        artifact_kind="trained_model",
        feature_names=["fp_0", "MolWt"],
        train_rows=8,
        test_rows=2,
        hyperparameters={"n_estimators": 10},
        metrics={"accuracy": 0.8},
        preprocessing={
            "imputer_strategy": "mean",
            "fit_scope": "training_partition_only",
            "persisted_in_model_pipeline": True,
        },
    )

    metadata_path = tmp_path / metadata_filename
    assert metadata_path.is_file()
    assert metadata["model_sha256"] == hashlib.sha256(b"model-payload").hexdigest()
    assert metadata["feature_count"] == 2
    assert metadata["feature_names"] == ["fp_0", "MolWt"]
    assert metadata["train_rows"] == 8
    assert metadata["preprocessing"]["fit_scope"] == "training_partition_only"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_training_task_saves_a_preprocessing_pipeline_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(qsarify_app.app.config, "MODEL_FOLDER", str(tmp_path))
    owner_id = "training-owner-" + "x" * 24
    task_id = "training-task-contract"
    X = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 40),
            "feature_b": [np.nan if index == 0 else float(index % 5) for index in range(40)],
        }
    )
    y = pd.Series([0, 1] * 20, name="bioactivity_class")
    qsarify_app.tasks[task_id] = {"status": "pending"}
    params = {
        "model_names": ["LogisticRegression"],
        "hyperparameters": {"LogisticRegression": {}},
        "test_size": 0.2,
        "use_resampling": False,
        "physchem": {"missing_strategy": "mean"},
    }

    qsarify_app.run_training_task(
        task_id,
        {"owner_id": owner_id, "processed_data": {"X": X.to_json(orient="split"), "y": y.to_json(orient="split")}},
        params,
    )

    task = qsarify_app.tasks.get(task_id)
    assert task["status"] == "complete"
    result = task["result"]["LogisticRegression"]
    model_path = Path(qsarify_app._get_owned_model_folder(owner_id)) / result["download_path"]
    saved_pipeline = qsarify_app.joblib.load(model_path)
    assert list(saved_pipeline.named_steps) == ["imputer", "model"]
    assert result["artifact_metadata"]["preprocessing"]["persisted_in_model_pipeline"] is True


def test_external_api_helper_retries_transient_failures_with_bounded_timeout(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise qsarify_app.requests.HTTPError(response=self)

    responses = iter([FakeResponse(503), FakeResponse(200)])
    calls = []
    sleeps = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return next(responses)

    monkeypatch.setattr(qsarify_app.requests, "get", fake_get)
    monkeypatch.setattr(qsarify_app.time, "sleep", lambda seconds: sleeps.append(seconds))

    response = qsarify_app._get_external_response("https://example.invalid/data", timeout=7)

    assert response.status_code == 200
    assert len(calls) == 2
    assert all(call[1]["timeout"] == 7 for call in calls)
    assert sleeps == [1]


def test_external_api_helper_does_not_retry_non_transient_http_errors(monkeypatch):
    class NotFoundResponse:
        status_code = 404

        def raise_for_status(self):
            raise qsarify_app.requests.HTTPError(response=self)

    calls = []
    monkeypatch.setattr(qsarify_app.requests, "get", lambda *args, **kwargs: calls.append(kwargs) or NotFoundResponse())

    with pytest.raises(qsarify_app.requests.HTTPError):
        qsarify_app._get_external_response("https://example.invalid/missing")

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({"uniprot_id": "bad!"}, "UniProt"),
        ({"activity_type": "IC50&limit=999"}, "activity type"),
        ({"threshold": "nan"}, "Threshold"),
        ({"threshold": "0"}, "Threshold"),
    ],
)
def test_chembl_stream_rejects_malformed_query_parameters(query, message):
    with qsarify_app.app.test_request_context("/api/chembl-data-stream", query_string=query):
        response = qsarify_app.get_chembl_data_stream()

    body, status = response
    assert status == 400
    assert message in body.get_json()["error"]


def test_training_reset_does_not_delete_shared_model_artifacts(tmp_path, monkeypatch):
    model_path = tmp_path / "model_shared.pkl"
    model_path.write_bytes(b"model")
    monkeypatch.setitem(qsarify_app.app.config, "MODEL_FOLDER", str(tmp_path))

    with qsarify_app.app.test_request_context("/api/training/reset", method="POST"):
        response = qsarify_app.reset_session()

    assert response.status_code == 200
    assert model_path.is_file()
    assert response.get_json()["models_deleted"] is False


def test_runtime_state_uses_configured_directories():
    source_dir = Path(qsarify_app.__file__).resolve().parent
    runtime_dir = Path(qsarify_app.RUNTIME_DIR).resolve()

    assert runtime_dir != source_dir
    assert qsarify_app.app.config["SESSION_TYPE"] == "cachelib"
    assert runtime_dir in Path(qsarify_app.app.config["SESSION_CACHE_DIR"]).resolve().parents
    assert runtime_dir in Path(qsarify_app.app.config["MODEL_FOLDER"]).resolve().parents
    assert runtime_dir in Path(qsarify_app.app.config["UPLOAD_FOLDER"]).resolve().parents
    assert runtime_dir in Path(qsarify_app.LOG_DIR).resolve().parents


def test_prediction_endpoint_returns_builtin_model_contract(monkeypatch):
    """Exercise the real Flask route and bundled model without changing defaults."""
    _require_bundled_model_fixture()
    previous_model = qsarify_app.loaded_model_data
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    try:
        assert qsarify_app.load_model_data() is True
        client = qsarify_app.app.test_client()
        response = client.post(
            "/predict",
            json={"smiles": "CCO", "target_protein": "AChE"},
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["smiles"] == "CCO"
        assert body["target_protein"] == "AChE"
        assert body["prediction"] in {"Active", "Inactive"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["applicability_domain"]["status"] in {
            AD_STATUS_IN_DOMAIN,
            AD_STATUS_BORDERLINE,
            AD_STATUS_OUT_OF_DOMAIN,
            AD_STATUS_UNAVAILABLE,
        }
        assert body["applicability_domain"]["reference_scope"] == "target_specific"
        assert body["applicability_domain"]["reference_target"] == "AChE"
    finally:
        qsarify_app.loaded_model_data = previous_model


def test_applicability_domain_uses_max_tanimoto_and_threshold_categories():
    query = np.asarray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("CCO"), 3, nBits=2048), dtype=np.uint8)
    packed = pack_fingerprint_matrix(np.asarray([query]))

    result = assess_molecule_applicability_domain(
        Chem.MolFromSmiles("CCO"),
        packed,
        np.asarray([int(query.sum())], dtype=np.int32),
    )
    assert result["max_similarity"] == 1.0
    assert result["status"] == AD_STATUS_IN_DOMAIN
    assert similarity_status(0.49) == AD_STATUS_BORDERLINE
    assert similarity_status(0.29) == AD_STATUS_OUT_OF_DOMAIN


def test_untrusted_model_uploads_cannot_be_enabled_in_flask():
    previous = qsarify_app.app.config.get("ALLOW_UNTRUSTED_MODEL_UPLOAD")
    qsarify_app.app.config["ALLOW_UNTRUSTED_MODEL_UPLOAD"] = True
    try:
        with qsarify_app.app.test_request_context("/ModelPlayground/predict", method="POST"):
            response, status = qsarify_app.ModelPlayground_predict()
    finally:
        qsarify_app.app.config["ALLOW_UNTRUSTED_MODEL_UPLOAD"] = previous

    assert status == 503
    assert response.get_json()["code"] == "isolated_model_worker_required"


def test_prediction_handler_rejects_missing_required_fields():
    with qsarify_app.app.test_request_context("/predict", method="POST", json={}):
        response = qsarify_app.predict()

    body, status = response
    assert status == 400
    assert "error" in body.get_json()


def test_prediction_handler_rejects_malformed_batch_items():
    with qsarify_app.app.test_request_context(
        "/predict", method="POST", json=[{"smiles": "CCO", "target_protein": "AChE"}, "not-an-object"]
    ):
        response = qsarify_app.predict()

    body, status = response
    assert status == 400
    assert body.get_json()["index"] == 1


def test_prediction_handler_rejects_invalid_target_before_model_call():
    with qsarify_app.app.test_request_context(
        "/predict", method="POST", json={"smiles": "CCO", "target_protein": "unknown"}
    ):
        response = qsarify_app.predict()

    body, status = response
    assert status == 400
    assert "Invalid target protein" in body.get_json()["error"]


def test_prediction_handler_rejects_oversized_smiles():
    oversized = "C" * (qsarify_app.MAX_PREDICTION_SMILES_LENGTH + 1)
    with qsarify_app.app.test_request_context(
        "/predict", method="POST", json={"smiles": oversized, "target_protein": "AChE"}
    ):
        response = qsarify_app.predict()

    body, status = response
    assert status == 413
    assert "maximum supported length" in body.get_json()["error"]


def test_training_upload_enforces_row_bound(monkeypatch):
    monkeypatch.setattr(qsarify_app, "MAX_TRAINING_UPLOAD_ROWS", 1)
    with qsarify_app.app.test_request_context(
        "/upload",
        method="POST",
        data={"files": (io.BytesIO(b"smiles,bioactivity_class\nCCO,Active\nCCN,Inactive\n"), "training.csv")},
        content_type="multipart/form-data",
    ):
        response = qsarify_app.upload_files()
    response_body, status = response

    assert status == 413
    assert "rows" in response_body.get_json()["error"]


def test_process_data_never_reuses_unowned_recent_uploads():
    with qsarify_app.app.test_request_context("/process-data", method="POST", json={}):
        response = qsarify_app.process_data()
    response_body, status = response

    assert status == 400
    assert "associated with this session" in response_body.get_json()["error"]


def test_process_data_reads_session_owned_upload_and_builds_features(tmp_path, monkeypatch):
    upload_path = tmp_path / "training.csv"
    upload_path.write_text("smiles,bioactivity_class\nCCO,Active\nCCN,Inactive\n", encoding="utf-8")
    monkeypatch.setitem(qsarify_app.app.config, "UPLOAD_FOLDER", str(tmp_path))

    with qsarify_app.app.test_request_context(
        "/process-data",
        method="POST",
        json={"use_fingerprints": {"enabled": True, "type": "Morgan", "radius": 2, "nbits": 8}},
    ):
        qsarify_app.session["uploaded_files"] = {"training.csv": {"path": str(upload_path)}}
        response = qsarify_app.process_data()

    assert response.status_code == 200
    body = response.get_json()
    assert body["summary"]["processed_shape"][0] == 2
    assert body["summary"]["processed_shape"][1] == 8
    assert body["data_id"]


@pytest.mark.parametrize("target", ["MAO-B", "COX-2", "VISFATIN", "BACE1", "AChE"])
def test_configured_alzheimer_targets_have_uniprot_mappings(target):
    assert target in qsarify_app.PROTEIN_MAP
    assert qsarify_app.PROTEIN_MAP[target]


def test_ache_mapping_is_acetylcholinesterase():
    assert qsarify_app.PROTEIN_MAP["AChE"] == "P22303"


def test_builtin_target_vector_matches_benchmark_one_hot_order():
    expected_names = sorted(qsarify_app.PROTEIN_MAP)
    expected_ids = [qsarify_app.PROTEIN_MAP[name] for name in expected_names]

    assert qsarify_app.TARGET_PROTEIN_NAMES_ORDERED == expected_names
    assert qsarify_app.PROTEIN_IDS_ORDERED == expected_ids


def test_benchmark_feature_contract_places_target_block_after_fingerprints():
    from experiments.run_benchmark import build_features

    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN", "CCCl"],
            "bioactivity_class": ["Active", "Inactive", "Active", "Inactive"],
            "target_name": ["AChE", "BACE1", "COX-2", "MAO-B"],
        }
    )
    X, _, metadata = build_features(
        frame,
        fingerprint_type="Morgan",
        fingerprint_radius=2,
        fingerprint_bits=16,
        descriptor_names=["MolWt"],
        one_hot_column="target_name",
        split_test_size=0.5,
        split_seed=42,
        split_strategy="random",
    )

    assert metadata["feature_order"] == ["fingerprints", "one_hot", "descriptors"]
    assert list(X.columns[:16]) == [f"fp_{i}" for i in range(16)]
    assert list(X.columns[-1:]) == ["MolWt"]


def test_builtin_model_metadata_matches_prediction_contract():
    _require_bundled_model_fixture()
    model_path = BUILTIN_MODEL_PATH
    metadata_path = BUILTIN_MODEL_METADATA_PATH

    assert model_path.is_file()
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model"]["n_features_in"] == 2058
    assert metadata["feature_metadata"]["feature_order"] == ["fingerprints", "one_hot", "descriptors"]
    assert metadata["feature_metadata"]["target_name_order"] == sorted(qsarify_app.PROTEIN_MAP)
    assert metadata["input"]["rows_read"] == 42374
    assert metadata["feature_metadata"]["row_count"] == 31193
    assert metadata["training_metadata"]["training_fingerprint_count"] == 31193
    assert set(metadata["training_metadata"]["training_fingerprint_by_target"]) == set(qsarify_app.PROTEIN_MAP)
    payload = joblib.load(model_path)
    assert set(payload["training_fingerprints_by_target"]) == set(qsarify_app.PROTEIN_MAP)
    assert metadata["feature_metadata"]["data_audit"]["invalid_activity_rows"] == 55
    assert metadata["feature_metadata"]["data_audit"]["duplicate_rows_collapsed"] == 11006
