#!/usr/bin/env python3
"""Bootstrap Tastytrade OAuth refresh token via the local dashboard."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

from aiohttp import ClientSession, ClientTimeout

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_secrets
from src.dashboard.web import ArgusWebDashboard


DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8777"


async def _check_dashboard(url: str) -> bool:
    timeout = ClientTimeout(total=2)
    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.get(f"{url}/api/status") as response:
                return response.status == 200
    except Exception:
        return False


def _has_refresh_token() -> bool:
    secrets = load_secrets()
    return bool(secrets.get("tastytrade_oauth2", {}).get("refresh_token"))


async def _wait_for_refresh_token(timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _has_refresh_token():
            return True
        await asyncio.sleep(2)
    return False


async def run_bootstrap(args: argparse.Namespace) -> int:
    if _has_refresh_token():
        print("Tastytrade refresh token already present in secrets.yaml.")
        return 0

    dashboard_url = args.dashboard_url.rstrip("/")
    dashboard_running = await _check_dashboard(dashboard_url)
    dashboard: Optional[ArgusWebDashboard] = None

    if not dashboard_running:
        dashboard = ArgusWebDashboard(host="127.0.0.1", port=8777)
        await dashboard.start()
        dashboard_url = f"http://{dashboard.host}:{dashboard.port}"
        print(f"Started local dashboard at {dashboard_url}")
    else:
        print(f"Using existing dashboard at {dashboard_url}")

    start_url = f"{dashboard_url}/oauth/tastytrade/start"
    print("Open the following URL to begin OAuth:")
    print(start_url)
    if args.open_browser:
        webbrowser.open(start_url)

    success = await _wait_for_refresh_token(args.timeout)
    if dashboard:
        await dashboard.stop()

    if success:
        print("Refresh token saved.")
        return 0
    print("Timed out waiting for refresh token.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Tastytrade OAuth refresh token.")
    parser.add_argument(
        "--dashboard-url",
        default=DEFAULT_DASHBOARD_URL,
        help="Dashboard base URL (default: http://127.0.0.1:8777).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds to wait for refresh token before exiting (default: 300).",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open the browser automatically.",
    )
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run_bootstrap(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
