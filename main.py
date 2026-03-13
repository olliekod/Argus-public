#!/usr/bin/env python3
"""
Argus - Crypto Market Monitor
=============================

Main entry point for running Argus.

Usage:
    python main.py
    python main.py --log-level DEBUG    # full tracebacks and debug lines (console + file)

To see debug logging (e.g. error tracebacks, bar-flush details):
  - Run:  python main.py --log-level DEBUG
  - Or set in config:  system.log_level: "DEBUG"
  - Or set env:  set ARGUS_LOG_LEVEL=DEBUG   (Windows)  /  export ARGUS_LOG_LEVEL=DEBUG   (Unix)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))


def _parse_args():
    p = argparse.ArgumentParser(description="Run Argus")
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Set log level (e.g. DEBUG for full error tracebacks and bar-flush diagnostics). Default from config.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.log_level:
        os.environ["ARGUS_LOG_LEVEL"] = args.log_level
    try:
        from src.orchestrator import main
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArgus stopped by user")
