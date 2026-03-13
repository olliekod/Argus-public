import shutil
import subprocess

import pytest


def test_secrets_yaml_not_tracked():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", "config/secrets.yaml"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        pytest.skip("git invocation failed")

    if result.returncode != 0:
        pytest.skip("git ls-files failed")

    assert result.stdout.strip() == ""
