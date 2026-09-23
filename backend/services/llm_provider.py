"""Real, server-side LLM provider adapters for Target Intelligence.

Providers are attempted only when their deployment credentials are present.
There are no mock providers or fabricated fallback responses.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import requests

from services.target_intelligence_config import get_runtime_settings, max_response_tokens, model_is_allowed


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, code: str = "provider_unavailable", status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


def _provider_order() -> list[str]:
    settings = get_runtime_settings()
    values = [settings.get("provider_primary"), *(settings.get("provider_fallbacks") or [])]
    return list(dict.fromkeys(value for value in values if value in {"openrouter", "openai", "azure", "anthropic"}))


def _credentials(provider: str) -> bool:
    if provider == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "azure":
        return bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


def _model_for(provider: str, requested_model: str) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_TARGET_INTELLIGENCE_MODEL", requested_model)
    if provider == "azure":
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT", requested_model)
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_TARGET_INTELLIGENCE_MODEL", requested_model)
    return requested_model


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    try:
        parsed = json.loads(content or "{}")
    except (TypeError, ValueError) as exc:
        raise ProviderError("Provider returned non-JSON structured content") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("Provider returned a non-object structured response")
    return parsed


def _call_openai_compatible(provider: str, model: str, system_prompt: str, user_prompt: str, schema: Mapping[str, Any], max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if provider == "openrouter":
        url = OPENROUTER_URL
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "HTTP-Referer": os.environ.get("NEXT_PUBLIC_SITE_URL", "http://localhost:5001"),
            "X-Title": "QSARify Target Intelligence",
        }
        provider_model = model
    elif provider == "openai":
        url = OPENAI_URL
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        provider_model = _model_for(provider, model)
    else:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY"]}
        provider_model = deployment
    response = requests.post(
        url,
        headers={**headers, "Content-Type": "application/json"},
        json={
            "model": provider_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "qsarify_structured_output", "strict": True, "schema": schema}},
        },
        timeout=90,
    )
    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            detail = str((body.get("error") or {}).get("message") or "") if isinstance(body, dict) else ""
        except (ValueError, TypeError):
            detail = ""
        lowered = detail.lower()
        code = "provider_budget_exceeded" if (
            "budget" in lowered
            or "credit" in lowered
            or "limit exceeded" in lowered
            or "rate limit" in lowered
            or response.status_code == 402
        ) else "provider_unavailable"
        status = 429 if code == "provider_budget_exceeded" else response.status_code
        suffix = f": {detail[:240]}" if detail else ""
        raise ProviderError(f"{provider} provider request failed with status {response.status_code}{suffix}", code=code, status=status)
    body = response.json()
    content = (body.get("choices") or [{}])[0].get("message", {}).get("content")
    return _parse_json_content(content), {"provider": provider, "model": provider_model, "usage": body.get("usage") or {}}


def _call_anthropic(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    provider_model = _model_for("anthropic", model)
    response = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": os.environ.get("ANTHROPIC_API_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
        json={
            "model": provider_model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise ProviderError(f"anthropic provider request failed with status {response.status_code}", status=response.status_code)
    body = response.json()
    content = body.get("content")
    usage = body.get("usage") or {}
    return _parse_json_content(content), {"provider": "anthropic", "model": provider_model, "usage": usage}


def call_structured(model: str, system_prompt: str, user_prompt: str, schema: Mapping[str, Any], max_tokens: int = 3000) -> tuple[dict[str, Any], dict[str, Any]]:
    if not model_is_allowed(model):
        raise ProviderError("Configured model is not allowed by Target Intelligence settings")
    bounded_tokens = min(max(int(max_tokens), 256), max_response_tokens())
    failures: list[str] = []
    failure_code = "provider_unavailable"
    failure_status: int | None = None
    for provider in _provider_order():
        if not _credentials(provider):
            continue
        try:
            if provider == "anthropic":
                return _call_anthropic(model, system_prompt, user_prompt, bounded_tokens)
            return _call_openai_compatible(provider, model, system_prompt, user_prompt, schema, bounded_tokens)
        except (ProviderError, requests.RequestException, ValueError) as exc:
            failures.append(f"{provider}: {str(exc)[:200]}")
            if isinstance(exc, ProviderError) and exc.code == "provider_budget_exceeded":
                failure_code = exc.code
                failure_status = exc.status
    if failures:
        raise ProviderError("All configured Target Intelligence providers failed: " + "; ".join(failures), code=failure_code, status=failure_status)
    raise ProviderError("No configured Target Intelligence provider credentials are available")
