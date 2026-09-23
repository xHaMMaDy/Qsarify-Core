from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer

import joblib
import pytest

from services.isolated_model_worker import _check_path
from services.isolated_model_worker_server import WorkerHandler
from services.workspace_prediction import predict_uploaded_features, validate_workspace_artifact


def test_isolated_worker_rejects_artifacts_outside_configured_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.joblib"
    outside.write_bytes(b"not a model")
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(root))
    with pytest.raises(ValueError, match="outside"):
        _check_path(str(outside))


def test_prediction_facade_does_not_deserialize_in_web_process():
    source = Path(__file__).resolve().parents[1] / "services" / "workspace_prediction.py"
    assert "joblib.load" not in source.read_text(encoding="utf-8")
    assert "isolated_model_worker" in source.read_text(encoding="utf-8")


def test_remote_worker_requires_token_and_validates_through_internal_server(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    artifact = root / "safe.joblib"
    joblib.dump({"profile": "fixture"}, artifact)
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("QSARIFY_MODEL_WORKER_TOKEN", "worker-fixture-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), WorkerHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("QSARIFY_MODEL_WORKER_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        result = validate_workspace_artifact(artifact_path=str(artifact))
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert result["model_type"] == "dict"
    assert result["artifact_size_bytes"] == artifact.stat().st_size


def test_uploaded_model_prediction_requires_exact_detected_feature_count(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    root = tmp_path / "models"
    root.mkdir()
    model = LogisticRegression().fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])
    artifact = root / "uploaded.joblib"
    joblib.dump(model, artifact)
    monkeypatch.setenv("QSARIFY_MODEL_ARTIFACT_ROOT", str(root))
    with pytest.raises(ValueError, match="exactly 2"):
        predict_uploaded_features(artifact_path=str(artifact), expected_sha256="", features=[[0.0]])
    result = predict_uploaded_features(artifact_path=str(artifact), expected_sha256="", features=[[0.0, 0.0], [1.0, 1.0]])
    assert len(result) == 2
    assert "confidence" in result[0]
