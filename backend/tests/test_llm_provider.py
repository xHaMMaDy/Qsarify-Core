import services.llm_provider as provider


def test_provider_adapter_uses_configured_openai_without_exposing_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-provider-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(provider, "get_runtime_settings", lambda: {"provider_primary": "openai", "provider_fallbacks": [], "allowed_models": [], "max_response_tokens": 1024})
    monkeypatch.setattr(provider, "model_is_allowed", lambda _model: True)
    monkeypatch.setattr(provider, "max_response_tokens", lambda: 1024)

    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    def fake_post(url, **kwargs):
        captured.update({"url": url, "kwargs": kwargs})
        return Response()

    monkeypatch.setattr(provider.requests, "post", fake_post)
    result, metadata = provider.call_structured("gpt-test", "system", "user", {"type": "object"}, max_tokens=800)

    assert result == {"ok": True}
    assert metadata["provider"] == "openai"
    assert captured["url"] == provider.OPENAI_URL
    assert "test-only-provider-key" in captured["kwargs"]["headers"]["Authorization"]
    assert captured["kwargs"]["json"]["max_tokens"] == 800


def test_provider_adapter_fails_closed_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(provider, "get_runtime_settings", lambda: {"provider_primary": "openrouter", "provider_fallbacks": [], "allowed_models": [], "max_response_tokens": 1024})
    monkeypatch.setattr(provider, "model_is_allowed", lambda _model: True)
    try:
        provider.call_structured("model", "system", "user", {"type": "object"})
    except provider.ProviderError as exc:
        assert "No configured" in str(exc)
    else:
        raise AssertionError("Expected provider adapter to fail closed")


def test_provider_budget_error_is_classified_without_leaking_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-provider-key")
    monkeypatch.setattr(provider, "get_runtime_settings", lambda: {"provider_primary": "openrouter", "provider_fallbacks": [], "allowed_models": [], "max_response_tokens": 1024})
    monkeypatch.setattr(provider, "model_is_allowed", lambda _model: True)
    monkeypatch.setattr(provider, "max_response_tokens", lambda: 1024)

    class Response:
        status_code = 403

        def json(self):
            return {"error": {"code": 403, "message": "Workspace daily budget exceeded"}}

    monkeypatch.setattr(provider.requests, "post", lambda *args, **kwargs: Response())
    try:
        provider.call_structured("model", "system", "user", {"type": "object"})
    except provider.ProviderError as exc:
        assert exc.code == "provider_budget_exceeded"
        assert exc.status == 429
        assert "test-only-provider-key" not in str(exc)
    else:
        raise AssertionError("Expected budget exhaustion to be classified")
