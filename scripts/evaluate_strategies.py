#!/usr/bin/env python3
"""
Strategy Evaluation CLI
========================

Loads experiment JSON outputs, computes composite rankings, and writes
a ranked strategy report.

Usage::

    python scripts/evaluate_strategies.py --input logs/experiments
    python scripts/evaluate_strategies.py --input logs/experiments --output logs/rankings.json
    python scripts/evaluate_strategies.py --input logs/experiments --quiet

The ranking JSON is written to ``logs/strategy_rankings_<date>.json``
by default.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# Ensure repo root is on sys.path so ``src`` imports work
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.strategy_evaluator import StrategyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and rank strategy experiment results.",
    )
    parser.add_argument(
        "--input",
        default="logs/experiments",
        help="Directory containing experiment JSON files (default: logs/experiments)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for rankings JSON. Default: logs/strategy_rankings_<date>.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress console summary table",
    )
    parser.add_argument(
        "--kill-thresholds",
        default=None,
        help="JSON string or YAML file path with kill thresholds",
    )
    parser.add_argument(
        "--output-killed",
        default=None,
        help="Optional output path for killed strategy list",
    )
    args = parser.parse_args()

    kill_thresholds = None
    if args.kill_thresholds:
        candidate = Path(args.kill_thresholds)
        if candidate.exists():
            with open(candidate, "r") as f:
                kill_thresholds = yaml.safe_load(f)
        else:
            kill_thresholds = json.loads(args.kill_thresholds)

    evaluator = StrategyEvaluator(input_dir=args.input, kill_thresholds=kill_thresholds)
    count = evaluator.load_experiments()

    if count == 0:
        print(f"No experiment files found in {args.input}")
        sys.exit(1)

    evaluator.evaluate()
    out_path = evaluator.save_rankings(output_path=args.output)

    if not args.quiet:
        evaluator.print_summary()

    if args.output_killed:
        killed_out = {
            "killed_count": len(evaluator.killed),
            "killed": evaluator.killed,
        }
        Path(args.output_killed).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_killed, "w") as f:
            json.dump(killed_out, f, indent=2)
        print(f"Killed list written to: {args.output_killed}")

    print(f"\nRankings written to: {out_path}")
    print(f"Evaluated {count} experiments.")


if __name__ == "__main__":
    main()
