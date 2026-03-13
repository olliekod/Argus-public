"""
Tests for secrets file permissions.

Verifies that save_secrets() sets mode 0o600 on the written file
so only the owner can read it.
"""

from __future__ import annotations

import importlib
import os
import stat
import sys
from pathlib import Path

import pytest

# Direct import to avoid heavy src.core.__init__ chain
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_config_mod = importlib.import_module("src.core.config")
save_secrets = _config_mod.save_secrets


class TestSecretsPermissions:
    def test_saved_secrets_have_mode_600(self, tmp_path, monkeypatch):
        """A written secrets file should have permissions 0o600 (Unix). On Windows, chmod(0o600) may not change st_mode."""
        secrets_path = str(tmp_path / "secrets.yaml")
        monkeypatch.setenv("ARGUS_SECRETS", secrets_path)

        save_secrets({"test_service": {"api_key": "test123"}})

        mode = os.stat(secrets_path).st_mode & 0o777
        if sys.platform == "win32":
            # Windows does not fully support Unix-style permission bits; st_mode often remains 0o666
            assert mode in (0o600, 0o666), f"Unexpected mode {oct(mode)}"
        else:
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_saved_secrets_readable_by_owner(self, tmp_path, monkeypatch):
        """Owner should be able to read the secrets file after writing."""
        secrets_path = str(tmp_path / "secrets.yaml")
        monkeypatch.setenv("ARGUS_SECRETS", secrets_path)

        save_secrets({"service": {"key": "value"}})

        # Verify the file is readable
        with open(secrets_path) as f:
            content = f.read()
        assert "key" in content
