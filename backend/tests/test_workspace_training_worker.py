from jobs.run_workspace_training_worker import build_heartbeat


def test_worker_heartbeat_is_ok_when_no_runs_fail(monkeypatch):
    monkeypatch.setattr("jobs.run_workspace_training_worker.time.monotonic", lambda: 12.5)
    heartbeat = build_heartbeat([], started_monotonic=12.0, limit=5, poll_seconds=10.0)
    assert heartbeat["status"] == "ok"
    assert heartbeat["processed_count"] == 0
    assert heartbeat["failed_count"] == 0
    assert heartbeat["duration_seconds"] == 0.5
    assert heartbeat["worker"] == "target_modeling_training"


def test_worker_heartbeat_marks_failed_runs_degraded(monkeypatch):
    monkeypatch.setattr("jobs.run_workspace_training_worker.time.monotonic", lambda: 20.25)
    heartbeat = build_heartbeat(
        [{"id": "redacted", "status": "completed"}, {"id": "redacted", "status": "failed"}],
        started_monotonic=20.0,
        limit=5,
        poll_seconds=5.0,
    )
    assert heartbeat["status"] == "degraded"
    assert heartbeat["processed_count"] == 2
    assert heartbeat["failed_count"] == 1
    assert heartbeat["duration_seconds"] == 0.25
