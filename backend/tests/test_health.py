from app import app


def test_health_endpoint_is_public_and_machine_readable():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "service": "qsarify-backend"}
