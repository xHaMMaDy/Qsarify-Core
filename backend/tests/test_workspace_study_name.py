import app as qsarify_app


def test_study_name_suggestion_is_editable_and_confirmation_gated(monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")

    def fake_call(_model, _system, _prompt, _schema, max_tokens=300):
        assert max_tokens == 300
        return {"suggested_name": "Breast Cancer Target QSAR Study", "rationale": "Concise target and disease context."}, {"provider": "fixture", "model": "fixture", "usage": {}}

    monkeypatch.setattr(qsarify_app, "call_structured", fake_call)
    response = qsarify_app.app.test_client().post("/api/target-intelligence/study-name", json={"target_name": "Fixture target"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["suggested_name"] == "Breast Cancer Target QSAR Study"
    assert body["requires_user_confirmation"] is True


def test_study_name_suggestion_rejects_empty_target(monkeypatch):
    monkeypatch.setattr(qsarify_app, "SUPABASE_URL", "")
    monkeypatch.setattr(qsarify_app, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(qsarify_app, "FLASK_ENV", "development")
    response = qsarify_app.app.test_client().post("/api/target-intelligence/study-name", json={"target_name": ""})
    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_request"
