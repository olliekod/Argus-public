"""Pre-commit guard to prevent committing secrets."""

from __future__ import annotations

import re
import subprocess
import sys


SUSPECT_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "COINGLASS_API_KEY",
    "OKX_API_KEY",
    "OKX_API_SECRET",
    "OKX_PASSPHRASE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TASTYTRADE_USERNAME",
    "TASTYTRADE_PASSWORD",
]

PLACEHOLDER_MARKERS = ("PASTE_", "YOUR_", "CHANGEME", "REPLACE_")


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _is_placeholder(value: str) -> bool:
    return any(marker in value for marker in PLACEHOLDER_MARKERS)


def main() -> int:
    staged_files = _run_git(["diff", "--cached", "--name-only"]).splitlines()
    if not staged_files:
        return 0

    if "config/secrets.yaml" in staged_files:
        print("ERROR: config/secrets.yaml is staged. Remove it before commit.")
        return 1

    pattern = re.compile("|".join(re.escape(key) for key in SUSPECT_KEYS), re.IGNORECASE)
    for path in staged_files:
        if path.endswith("config/secrets.example.yaml"):
            continue
        blob = _run_git(["show", f":{path}"])
        if not blob:
            continue
        if "\x00" in blob:
            continue
        for line in blob.splitlines():
            if pattern.search(line):
                value = line.split(":", 1)[1] if ":" in line else line
                if not _is_placeholder(value):
                    print(f"ERROR: Possible secret detected in {path}.")
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
