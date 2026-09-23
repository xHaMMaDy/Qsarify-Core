"""Process queued Target Modeling Workspace training runs once.

Run this from a scheduler or a long-running worker supervisor. It uses the
same atomic database claim as the web-triggered beta worker and is safe to run
more than once.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from services.workspace_training_jobs import process_pending_training_runs


def build_heartbeat(results: list[Mapping[str, Any]], *, started_monotonic: float, limit: int, poll_seconds: float) -> dict[str, Any]:
    """Return scheduler-friendly health metadata without exposing run data."""
    failed = sum(1 for result in results if result.get("status") in {"failed", "error"} or result.get("error"))
    return {
        "status": "degraded" if failed else "ok",
        "worker": "target_modeling_training",
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "processed_count": len(results),
        "failed_count": failed,
        "duration_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "limit": limit,
        "poll_seconds": poll_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--loop", action="store_true", help="Poll continuously until interrupted")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    poll_seconds = min(max(args.poll_seconds, 1.0), 300.0)
    while True:
        started_monotonic = time.monotonic()
        results = process_pending_training_runs(args.limit)
        heartbeat = build_heartbeat(results, started_monotonic=started_monotonic, limit=args.limit, poll_seconds=poll_seconds)
        print(json.dumps({**heartbeat, "processed": results}, default=str), flush=True)
        if not args.loop:
            return 1 if heartbeat["failed_count"] else 0
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
