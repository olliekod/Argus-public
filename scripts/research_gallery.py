#!/usr/bin/env python3
# Created by Oliver Meihls

# Research Gallery & Strategy Bank CLI
#
# Allows listing, viewing, and batch-backtesting strategies stored in
# the factory database (Athena's promoted bank).
#
# Usage::
#
# # List all Gold/Silver strategies
# python scripts/research_gallery.py list
#
# # View details for a specific case
# python scripts/research_gallery.py view [case_id]
#
# # Batch test all promoted but un-backtested strategies
# python scripts/research_gallery.py run-untested

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.agent.pantheon.factory import FactoryPipe
from scripts.strategy_research_loop import run_cycle
from src.analysis.research_loop_config import load_research_loop_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.gallery")

def list_strategies(pipe: FactoryPipe):
    # List all promoted strategies.
    promoted = pipe.get_promoted_strategies()
    if not promoted:
        print("No promoted strategies found in factory.db")
        return

    print(f"\n{'='*90}")
    print(f"{'Case ID':<30} | {'Grading':<8} | {'Name'}")
    print(f"{'-'*90}")
    for s in promoted:
        print(f"{s['case_id']:<30} | {s['grading']:<8} | {s['name']}")
    print(f"{'='*90}")
    print(f"Total: {len(promoted)} promoted strategies.")

def view_case(pipe: FactoryPipe, case_id: str):
    # View details for a specific case.
    case = pipe.get_case(case_id)
    if not case:
        print(f"Case {case_id} not found.")
        return

    s = case["strategy"]
    print(f"\n[ Strategy: {s['name']} ]")
    print(f"Case ID: {s['case_id']}")
    print(f"Grading: {s['grading']}")
    print(f"Status:  {s['status']}")
    print(f"Athena Confidence: {s['athena_confidence']:.2f}")
    print("\n[ Final Manifest ]")
    if s["final_manifest"]:
        print(json.dumps(json.loads(s["final_manifest"]), indent=2))
    else:
        print("No manifest found.")

def run_untested(pipe: FactoryPipe):
    # Run backtests for all promoted strategies that haven't been run yet.
    promoted = pipe.get_promoted_strategies()
    if not promoted:
        print("No promoted strategies to run.")
        return

    config_path = "config/research_loop.yaml"
    try:
        config = load_research_loop_config(config_path)
    except Exception as e:
        logger.error("Failed to load research loop config: %s", e)
        return

    print(f"Found {len(promoted)} promoted strategies. Starting batch run...")
    for s in promoted:
        case_id = s["case_id"]
        logger.info("[Gallery] Starting backtest for %s (%s)...", case_id, s["name"])
        try:
            run_cycle(config, case_id=case_id)
        except Exception as e:
            logger.error("Failed to run backtest for %s: %s", case_id, e)

def main():
    parser = argparse.ArgumentParser(description="Argus Research Gallery")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("list", help="List promoted strategies")
    
    view_parser = subparsers.add_parser("view", help="View a specific case")
    view_parser.add_argument("case_id", help="The case_id to view")

    subparsers.add_parser("run-untested", help="Run backtests for all promoted strategies")

    args = parser.parse_args()
    pipe = FactoryPipe()

    if args.command == "list":
        list_strategies(pipe)
    elif args.command == "view":
        view_case(pipe, args.case_id)
    elif args.command == "run-untested":
        run_untested(pipe)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
