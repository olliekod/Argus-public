from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DWARF_NAMES_PATH = ROOT / "argus_kalshi" / "dwarf_names.txt"
LEGACY_NAME_COUNT = 7488
LEGACY_PREFIX_SHA256 = "4bbbffc03c6935d5e2096ec4831532de2118f1344d523c41dc6069e2b78d8769"
LEGACY_FIRST_10 = [
    "Balabur",
    "Balagrom",
    "Balaiain",
    "Balaiar",
    "Balaidain",
    "Balaikor",
    "Balaimor",
    "Balairon",
    "Balaivar",
    "Balakorn",
]
LEGACY_LAST_10 = [
    "Zulunor",
    "Zulunskar",
    "Zulunzorn",
    "Zulurbur",
    "Zulurnar",
    "Zulurzul",
    "Zuluthrum",
    "Zuluur",
    "Zuluvek",
    "Zuluvorn",
]


def test_dwarf_names_file_has_10000_lines() -> None:
    lines = DWARF_NAMES_PATH.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 10000


def test_dwarf_names_preserve_legacy_prefix() -> None:
    names = DWARF_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    legacy_prefix = names[:LEGACY_NAME_COUNT]
    legacy_hash = hashlib.sha256("\n".join(legacy_prefix).encode("utf-8")).hexdigest()

    assert legacy_prefix[:10] == LEGACY_FIRST_10
    assert legacy_prefix[-10:] == LEGACY_LAST_10
    assert legacy_hash == LEGACY_PREFIX_SHA256
