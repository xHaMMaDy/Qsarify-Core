"""Evaluate source-backed Target Intelligence benchmark outputs.

Usage:
  python backend/experiments/evaluate_target_intelligence_benchmark.py \
    --cases docs/target-intelligence/benchmark_cases.csv \
    --reports artifacts/target-intelligence-reports.jsonl \
    --output artifacts/target-intelligence-metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.target_intelligence_benchmark import BenchmarkInputError, evaluate_benchmark, load_gold_cases, load_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = evaluate_benchmark(load_gold_cases(args.cases), load_reports(args.reports))
    except BenchmarkInputError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases_evaluated": result["cases_evaluated"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
