import io
import uuid

import app as qsarify_app


class FakeStore:
    def __init__(self, row):
        self.row = row

    def request(self, method, path, *, params=None, payload=None):
        if method == "GET" and path == "ti_uploaded_models":
            if params and params.get("id") == f"eq.{self.row['id']}":
                return [self.row]
            return []
        raise AssertionError(f"Unexpected {method} {path}")


def _auth_context(monkeypatch, path, method="POST", **kwargs):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    return qsarify_app.app.test_request_context(path, method=method, **kwargs)


def test_uploaded_model_route_stores_only_after_worker_inspection(monkeypatch, tmp_path):
    owner_id = str(uuid.uuid4())
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(qsarify_app, "validate_workspace_artifact", lambda **_kwargs: {"model_type": "LogisticRegression", "feature_count": 2, "has_predict": True, "has_predict_proba": True})
    with _auth_context(monkeypatch, "/api/target-intelligence/model-upload", content_type="multipart/form-data", data={"model": (io.BytesIO(b"safe-fixture"), "model.joblib"), "display_name": "Fixture model"}):
        qsarify_app.request.user = {"sub": owner_id}
        response = qsarify_app.target_intelligence_model_upload()
    assert response[1] == 201
    body = response[0].get_json()
    assert body["model"]["metadata_confirmed"] is False
    assert body["model"]["capability_profile"]["feature_count"] == 2


def test_uploaded_model_prediction_requires_metadata_confirmation(monkeypatch):
    owner_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    row = {"id": model_id, "user_id": owner_id, "artifact_format": "joblib", "checksum_sha256": "a" * 64, "metadata_confirmed": False, "status": "ready"}
    monkeypatch.setattr(qsarify_app, "_workspace_store", lambda: FakeStore(row))
    with _auth_context(monkeypatch, "/api/target-intelligence/uploaded-model-predict", content_type="application/json", json={"model_id": model_id, "features": [[1, 2]]}):
        qsarify_app.request.user = {"sub": owner_id}
        response = qsarify_app.target_intelligence_uploaded_model_predict()
    assert response[1] == 409
    assert response[0].get_json()["code"] == "metadata_confirmation_required"
