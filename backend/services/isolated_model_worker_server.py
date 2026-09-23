"""Internal HTTP adapter for the isolated model-worker container.

The service accepts only bounded, authenticated JSON from the backend service.
Its container is mounted read-only and should be placed on an internal-only
network with no egress in production.
"""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .isolated_model_worker import _predict, _predict_features, _validate


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "QSARifyModelWorker/1"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/execute":
            self._reply(404, {"error": "not_found"})
            return
        configured = os.environ.get("QSARIFY_MODEL_WORKER_TOKEN", "")
        supplied = self.headers.get("X-QSARIFY-Worker-Token", "")
        if not configured or not hmac.compare_digest(supplied, configured):
            self._reply(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16 * 1024 * 1024:
                raise ValueError("Worker request exceeds the 16 MB limit")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Worker request must be an object")
            result = _validate(payload) if payload.get("operation") == "validate" else _predict(payload) if payload.get("operation") == "predict" else _predict_features(payload) if payload.get("operation") == "predict_features" else None
            if result is None:
                raise ValueError("Unsupported worker operation")
            self._reply(200, result)
        except Exception as exc:
            self._reply(400, {"error": str(exc)[:500]})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._reply(200, {"status": "ok", "worker": "isolated_model"})
            return
        self._reply(404, {"error": "not_found"})

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    host = os.environ.get("QSARIFY_MODEL_WORKER_HOST", "127.0.0.1")
    port = int(os.environ.get("QSARIFY_MODEL_WORKER_PORT", "5060"))
    ThreadingHTTPServer((host, port), WorkerHandler).serve_forever()


if __name__ == "__main__":
    main()
