from pathlib import Path
import re


def test_no_orchestrator_variants_exist():
    repo_root = Path(__file__).resolve().parents[1]
    variants = [
        path for path in repo_root.rglob("orchestrator*.py")
        if path.as_posix().endswith("src/orchestrator.py") is False
        and path.name != "orchestrator.py"
    ]
    assert not variants, f"Unexpected orchestrator variants found: {variants}"


def test_no_legacy_orchestrator_imports():
    repo_root = Path(__file__).resolve().parents[1]
    forbidden_names = {
        "orchestrator_LOCAL",
        "orchestrator_REMOTE",
        "orchestrator_local_old",
    }
    import_patterns = [
        re.compile(r"from\s+src\.orchestrator_", re.IGNORECASE),
        re.compile(r"import\s+src\.orchestrator_", re.IGNORECASE),
    ]

    offenders = []
    for path in repo_root.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(name in text for name in forbidden_names):
            offenders.append(path)
            continue
        if any(pattern.search(text) for pattern in import_patterns):
            offenders.append(path)
    assert not offenders, f"Legacy orchestrator imports found: {offenders}"
