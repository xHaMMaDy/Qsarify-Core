import uuid

import app as qsarify_app


class FakeStore:
    def __init__(self, deployment):
        self.deployment = deployment

    def request(self, method, path, *, params=None, payload=None):
        if method == "GET" and path == "ti_model_deployments":
            if self.deployment and self.deployment.get("allow_public_download") is True and params and params.get("id") == f"eq.{self.deployment['id']}" and params.get("allow_public_download") == "eq.true":
                return [self.deployment]
            return []
        if method == "GET" and path == "ti_model_artifacts":
            if self.deployment and params and params.get("id") == f"eq.{self.deployment['artifact_id']}":
                return [{"id": self.deployment["artifact_id"], "user_id": self.deployment["user_id"], "artifact_path": "fixture.joblib", "status": "deployed"}]
            return []
        raise AssertionError(f"Unexpected {method} {path}")


def test_public_artifact_route_fails_closed_without_internal_token(monkeypatch):
    deployment_id = str(uuid.uuid4())
    monkeypatch.setenv("QSARIFY_PUBLIC_INTERNAL_TOKEN", "public-fixture-token")
    with qsarify_app.app.test_request_context(f"/api/target-intelligence/public-artifact-download?deployment_id={deployment_id}"):
        response = qsarify_app.target_intelligence_public_artifact_download()
    assert response[1] == 404
    assert response[0].get_json()["code"] == "public_artifact_not_found"


def test_public_artifact_route_serves_only_an_enabled_model(monkeypatch, tmp_path):
    deployment_id = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())
    owner_id = str(uuid.uuid4())
    artifact = tmp_path / "fixture.joblib"
    artifact.write_bytes(b"public-fixture-model")
    monkeypatch.setenv("QSARIFY_PUBLIC_INTERNAL_TOKEN", "public-fixture-token")
    monkeypatch.setattr(qsarify_app, "_workspace_store", lambda: FakeStore({"id": deployment_id, "artifact_id": artifact_id, "user_id": owner_id, "allow_public_download": True}))
    monkeypatch.setattr(qsarify_app, "_workspace_artifact_path", lambda _relative: artifact)
    response = qsarify_app.app.test_client().get(f"/api/target-intelligence/public-artifact-download?deployment_id={deployment_id}&kind=model", headers={"X-QSARIFY-Public-Token": "public-fixture-token"})
    assert response.status_code == 200
    assert response.data == b"public-fixture-model"


def test_public_artifact_route_requires_owner_enabled_policy(monkeypatch):
    deployment_id = str(uuid.uuid4())
    monkeypatch.setenv("QSARIFY_PUBLIC_INTERNAL_TOKEN", "public-fixture-token")
    monkeypatch.setattr(qsarify_app, "_workspace_store", lambda: FakeStore({"id": deployment_id}))
    with qsarify_app.app.test_request_context(f"/api/target-intelligence/public-artifact-download?deployment_id={deployment_id}", headers={"X-QSARIFY-Public-Token": "public-fixture-token"}):
        response = qsarify_app.target_intelligence_public_artifact_download()
    assert response[1] == 404
    assert response[0].get_json()["code"] == "public_artifact_not_found"
