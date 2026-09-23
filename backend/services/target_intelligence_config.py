"""Server-only Target Intelligence runtime configuration.

The browser never receives these values. Supabase settings are cached briefly
to avoid adding a database request to every provider call; safe environment
defaults keep local development functional when the settings table is absent.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests


_cache: dict[str, Any] = {"checked_at": 0.0, "settings": None}
_CACHE_SECONDS = 30.0


def get_runtime_settings() -> dict[str, Any]:
    now = time.time()
    if isinstance(_cache.get("settings"), dict) and now - float(_cache.get("checked_at") or 0) < _CACHE_SECONDS:
        return dict(_cache["settings"])

    defaults = {
        "max_response_tokens": int(os.environ.get("TARGET_INTELLIGENCE_MAX_RESPONSE_TOKENS", "4096")),
        "allowed_models": [],
        "provider_primary": "openrouter",
        "provider_fallbacks": ["openai", "azure", "anthropic"],
    }
    base_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if base_url and service_key:
        try:
            response = requests.get(
                f"{base_url}/rest/v1/ti_admin_settings",
                params={
                    "select": "max_response_tokens,allowed_models,provider_primary,provider_fallbacks",
                    "setting_key": "eq.global",
                    "limit": 1,
                },
                headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
                timeout=5,
            )
            response.raise_for_status()
            row = (response.json() or [{}])[0]
            if isinstance(row, dict):
                defaults.update({key: row[key] for key in defaults if key in row})
        except (requests.RequestException, TypeError, ValueError):
            pass
    try:
        defaults["max_response_tokens"] = min(max(int(defaults["max_response_tokens"]), 256), 32768)
    except (TypeError, ValueError):
        defaults["max_response_tokens"] = 4096
    defaults["allowed_models"] = [str(model) for model in defaults.get("allowed_models", []) if isinstance(model, str) and model.strip()]
    _cache.update({"checked_at": now, "settings": defaults})
    return dict(defaults)


def model_is_allowed(model: str) -> bool:
    allowed = get_runtime_settings().get("allowed_models") or []
    return not allowed or model in allowed


def max_response_tokens() -> int:
    return int(get_runtime_settings().get("max_response_tokens", 4096))
