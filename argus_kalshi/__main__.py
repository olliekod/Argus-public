"""
CLI entrypoint for the Argus Kalshi trading system.

    python -m argus_kalshi

Runs with the compact farm config by default (params generated from grid, no drift).
Override with --settings / ARGUS_SETTINGS_PATH and --secrets / ARGUS_SECRETS_PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="argus_kalshi",
        description="Argus Kalshi BTC strike contract trading system",
    )
    parser.add_argument(
        "--settings",
        default=os.environ.get("ARGUS_SETTINGS_PATH", "config/kalshi_farm_compact.yaml"),
        help="Path to settings YAML (env: ARGUS_SETTINGS_PATH)",
    )
    parser.add_argument(
        "--secrets",
        default=os.environ.get("ARGUS_SECRETS_PATH", "config/secrets.yaml"),
        help="Path to secrets YAML (env: ARGUS_SECRETS_PATH)",
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Run only the terminal UI client (connect to trading process IPC)",
    )
    parser.add_argument(
        "--connect",
        default="127.0.0.1:9999",
        help="Host:port for IPC when --ui-only (default: 127.0.0.1:9999)",
    )
    args = parser.parse_args()

    if args.ui_only:
        from .ui_client import main as ui_main
        ui_main(connect=args.connect)
        return

    from .runner import run

    try:
        asyncio.run(run(args.settings, args.secrets))
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        sys.exit(1)


main()
