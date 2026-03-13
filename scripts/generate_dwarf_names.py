"""
Generate legacy-stable LOTR dwarf-style names for Kalshi farm bots.

Outputs exactly 10,000 unique names by default to `argus_kalshi/dwarf_names.txt`.
The first 7,488 names are preserved from the existing checked-in file so
historical bot IDs remain stable, then additional names are appended from the
prefix × suffix pool in deterministic order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_COUNT = 10000
LEGACY_NAME_COUNT = 7488
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "argus_kalshi" / "dwarf_names.txt"

# Syllable blocks that sound Tolkien-dwarf (Khuzdul / Norse-inspired).
# Existing entries stay in-place so generated extension order remains stable.
PREFIXES = [
    "Bal", "Bif", "Bor", "Dor", "Dur", "Dwal", "Fili", "Fror", "Fund", "Gim",
    "Glin", "Gor", "Grim", "Har", "Hath", "Kili", "Kor", "Lor", "Mor", "Nar",
    "Nor", "Ori", "Orn", "Rur", "Thor", "Thra", "Tor", "Var", "Bral", "Brin",
    "Dral", "Drar", "Fral", "Gral", "Kral", "Mral", "Nral", "Thral", "Vral",
    "Brom", "Drom", "From", "Grom", "Krom", "Mrom", "Trom", "Vrom", "Bran",
    "Dran", "Gran", "Kran", "Mran", "Tran", "Vran", "Brod", "Drod", "Grod",
    "Krod", "Mrod", "Trod", "Bond", "Fond", "Gond", "Kond", "Mond", "Rond",
    "Tond", "Vond", "Bund", "Dund", "Gund", "Kund", "Mund", "Rund", "Tund",
    # Extended prefixes — added for 10k+ bot support
    "Azur", "Baur", "Caur", "Daur", "Faur", "Gaur", "Haur", "Kaur", "Maur", "Naur",
    "Raur", "Taur", "Vaur", "Wuld", "Xuld", "Yuld", "Zuld", "Brath", "Drath", "Frath",
    "Grath", "Krath", "Mrath", "Nrath", "Trath", "Vrath", "Brond", "Drond", "Frond",
    "Grond", "Krond", "Mrond", "Trond", "Vrond", "Brund", "Drund", "Frund", "Grund",
    "Krund", "Mrund", "Trund", "Vrund",
]

# Many suffixes so we have comfortable headroom with no numeric suffixes.
SUFFIXES = [
    "ain", "ar", "bur", "dain", "din", "dir", "drin", "far", "gorn", "grim",
    "im", "in", "inor", "li", "mir", "nar", "or", "orn", "rak", "rum", "thor", "ur",
    "al", "an", "ak", "am", "ath", "ek", "el", "en", "il", "ir", "om", "on", "uk", "um", "un",
    "bor", "dor", "dur", "for", "gor", "kor", "mor", "nor", "tor", "vor",
    "bin", "fin", "gin", "kin", "min", "nin", "rin", "tin", "vin",
    "bal", "dal", "fal", "gal", "kal", "mal", "nal", "ral", "tal", "val",
    "ban", "dan", "fan", "gan", "kan", "man", "nan", "ran", "tan", "van",
    "bir", "dir", "fir", "gir", "kir", "mir", "nir", "rir", "tir", "vir",
    "bak", "dak", "fak", "gak", "kak", "mak", "nak", "rak", "tak", "vak",
    "bol", "dol", "fol", "gol", "kol", "mol", "nol", "rol", "tol", "vol",
    "bon", "don", "fon", "gon", "kon", "mon", "non", "ron", "ton", "von",
    "bun", "dun", "fun", "gun", "kun", "mun", "nun", "run", "tun", "vun",
    "bar", "dar", "gar", "kar", "mar", "sar", "tar", "var",
    "bel", "del", "fel", "gel", "kel", "mel", "nel", "rel", "tel", "vel",
    "ben", "den", "fen", "gen", "ken", "men", "nen", "ren", "ten", "ven",
    "bil", "dil", "fil", "gil", "kil", "mil", "nil", "ril", "til", "vil",
    "bul", "dul", "ful", "gul", "kul", "mul", "nul", "rul", "tul", "vul",
    "bor", "dol", "fol", "gol", "kol", "mol", "rol", "tol",
    "bath", "dath", "gath", "kath", "math", "nath", "rath", "tath",
    "bek", "dek", "gek", "kek", "mek", "nek", "rek", "tek", "vek",
]


def load_legacy_names(path: Path, limit: int = LEGACY_NAME_COUNT) -> list[str]:
    """Load the legacy checked-in bot name prefix so existing bot IDs stay stable."""
    if not path.exists():
        return []
    names = path.read_text(encoding="utf-8").split()
    return names[:limit]


def generate_names(count: int = DEFAULT_COUNT, legacy_names: list[str] | None = None) -> list[str]:
    """Produce `count` names while preserving the legacy 7,488-name prefix."""
    if legacy_names is None:
        legacy_names = load_legacy_names(DEFAULT_OUTPUT)

    out: list[str] = []
    seen: set[str] = set()

    if legacy_names:
        for name in legacy_names[:count]:
            out.append(name)
            seen.add(name)

    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            name = prefix + suffix
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
            if len(out) >= count:
                return out[:count]

    # Fallback only if a caller asks for more names than the unique pool supports.
    while len(out) < count and out:
        out.extend(out[: min(len(out), count - len(out))])
    return out[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 10,000 legacy-stable LOTR dwarf-style bot names")
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of names to generate (default 10000)",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(DEFAULT_OUTPUT.relative_to(ROOT)),
        help="Output file path (default argus_kalshi/dwarf_names.txt for 10,000 names)",
    )
    args = parser.parse_args()

    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT / path
    legacy_source = path if path.exists() else DEFAULT_OUTPUT
    legacy_names = load_legacy_names(legacy_source)
    names = generate_names(args.count, legacy_names=legacy_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {len(names)} names to {path}")


if __name__ == "__main__":
    main()
