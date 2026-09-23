"""Run pending Target Intelligence jobs against real Supabase/public APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.target_intelligence_jobs import SupabaseWorkerStore, process_pending_runs, process_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    store = SupabaseWorkerStore()
    if args.run_id:
        result = process_run(args.run_id, store=store)
    else:
        result = process_pending_runs(args.limit, store=store)
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
