"""
Tests for secrets file permissions guard (10.8).

Verifies:
- POSIX: detects world-readable and fixes to 0o600
- POSIX: leaves safe permissions alone
- POSIX: warns (does not break) if chmod fails
- Windows: warns with guidance, no enforcement
- Non-existent file: skips gracefully

Uses mocked os/stat calls — no real filesystem permission changes.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.secrets_guard import check_secrets_permissions, _SAFE_MODE, _GROUP_OTHER_MASK


class FakeStat:
    """Simulate os.stat result with a configurable mode."""

    def __init__(self, mode: int):
        self.st_mode = mode | stat.S_IFREG  # Regular file


def _make_stat(mode: int):
    """Create a stat function that returns a FakeStat with given mode."""
    def fake_stat(path):
        return FakeStat(mode)
    return fake_stat


class TestPosixSafe:
    """File already has 0o600 — should report 'ok'."""

    def test_safe_permissions(self, tmp_path):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        result = check_secrets_permissions(
            path=path,
            enforce=True,
            _stat_fn=_make_stat(0o600),
            _chmod_fn=MagicMock(),
            _platform_fn=lambda: "Linux",
        )
        assert result["exists"] is True
        assert result["is_safe"] is True
        assert result["action"] == "ok"
        assert result["platform"] == "posix"


class TestPosixUnsafeFixed:
    """File is 0o644 — should be fixed to 0o600."""

    def test_fixes_world_readable(self, tmp_path):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        mock_chmod = MagicMock()

        result = check_secrets_permissions(
            path=path,
            enforce=True,
            _stat_fn=_make_stat(0o644),
            _chmod_fn=mock_chmod,
            _platform_fn=lambda: "Linux",
        )
        assert result["exists"] is True
        assert result["is_safe"] is False
        assert result["action"] == "fixed"
        mock_chmod.assert_called_once_with(str(path), _SAFE_MODE)


class TestPosixChmodFails:
    """chmod fails (e.g., not owner) — should warn, not crash."""

    def test_warns_on_chmod_failure(self, tmp_path):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        def failing_chmod(p, mode):
            raise OSError("Permission denied")

        result = check_secrets_permissions(
            path=path,
            enforce=True,
            _stat_fn=_make_stat(0o644),
            _chmod_fn=failing_chmod,
            _platform_fn=lambda: "Linux",
        )
        assert result["action"] == "warning"
        assert "Could not chmod" in result["message"]


class TestPosixNoEnforce:
    """enforce=False: warns but does not chmod."""

    def test_no_enforce_warns_only(self, tmp_path):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        mock_chmod = MagicMock()

        result = check_secrets_permissions(
            path=path,
            enforce=False,
            _stat_fn=_make_stat(0o644),
            _chmod_fn=mock_chmod,
            _platform_fn=lambda: "Linux",
        )
        assert result["action"] == "warning"
        mock_chmod.assert_not_called()


class TestWindows:
    """On Windows: warn with guidance, no chmod."""

    def test_windows_warning(self, tmp_path):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        result = check_secrets_permissions(
            path=path,
            enforce=True,
            _platform_fn=lambda: "Windows",
        )
        assert result["platform"] == "windows"
        assert result["action"] == "warning"
        assert "Right-click" in result["message"]


class TestNonExistent:
    """File does not exist — skip gracefully."""

    def test_missing_file(self, tmp_path):
        path = tmp_path / "does_not_exist.yaml"

        result = check_secrets_permissions(
            path=path,
            _platform_fn=lambda: "Linux",
        )
        assert result["exists"] is False
        assert result["action"] == "skipped"


class TestGroupWritableBits:
    """Verify the _GROUP_OTHER_MASK catches various unsafe modes."""

    @pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o666, 0o755])
    def test_unsafe_modes_detected(self, tmp_path, mode):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        result = check_secrets_permissions(
            path=path,
            enforce=False,
            _stat_fn=_make_stat(mode),
            _chmod_fn=MagicMock(),
            _platform_fn=lambda: "Linux",
        )
        assert result["is_safe"] is False

    @pytest.mark.parametrize("mode", [0o600, 0o400, 0o200])
    def test_safe_modes_pass(self, tmp_path, mode):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        result = check_secrets_permissions(
            path=path,
            enforce=False,
            _stat_fn=_make_stat(mode),
            _chmod_fn=MagicMock(),
            _platform_fn=lambda: "Linux",
        )
        assert result["is_safe"] is True


class TestDarwin:
    """macOS should be treated as POSIX."""

    def test_darwin_is_posix(self, tmp_path):
        path = tmp_path / "secrets.yaml"
        path.write_text("key: value")

        result = check_secrets_permissions(
            path=path,
            enforce=False,
            _stat_fn=_make_stat(0o600),
            _chmod_fn=MagicMock(),
            _platform_fn=lambda: "Darwin",
        )
        assert result["platform"] == "posix"
        assert result["is_safe"] is True
