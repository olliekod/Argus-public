"""
Secrets file permission guard (10.8).

Cross-platform helper that warns or enforces that secrets.yaml is not
world-readable.

- POSIX: chmod to 0o600 if too open (best-effort, does not break if
  the chmod fails).
- Windows: log a warning with guidance (no enforcement — ACLs are too
  complex for a generic helper).
"""

from __future__ import annotations

import logging
import os
import platform
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Permission mask for "group or other" read/write/execute bits
_GROUP_OTHER_MASK = (
    stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
    stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH
)

# Target permission: owner read+write only
_SAFE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600


def check_secrets_permissions(
    path: Optional[Path] = None,
    *,
    enforce: bool = True,
    _stat_fn=None,
    _chmod_fn=None,
    _platform_fn=None,
) -> dict:
    """Check (and optionally enforce) that a secrets file is not world-readable.

    Parameters
    ----------
    path : Path, optional
        Path to the secrets file.  Defaults to ``config/secrets.yaml``
        relative to the repo root.
    enforce : bool
        If True (default) and on POSIX, attempt to chmod the file to 0o600.
    _stat_fn, _chmod_fn, _platform_fn :
        Injectable overrides for testing without touching the real filesystem.

    Returns
    -------
    dict with keys:
        - ``path``: str — resolved path
        - ``exists``: bool
        - ``platform``: str — "posix" or "windows" or "unknown"
        - ``mode_octal``: str | None — e.g. "0o644"
        - ``is_safe``: bool — True if no group/other bits set
        - ``action``: str — "ok", "fixed", "warning", "skipped", "error"
        - ``message``: str — human-readable summary
    """
    stat_fn = _stat_fn or os.stat
    chmod_fn = _chmod_fn or os.chmod
    platform_fn = _platform_fn or platform.system

    if path is None:
        from src.core.config import find_repo_root
        path = find_repo_root() / "config" / "secrets.yaml"

    result = {
        "path": str(path),
        "exists": False,
        "platform": "unknown",
        "mode_octal": None,
        "is_safe": False,
        "action": "skipped",
        "message": "",
    }

    # Determine platform
    sys_name = platform_fn().lower()
    if sys_name in ("linux", "darwin", "freebsd"):
        result["platform"] = "posix"
    elif sys_name == "windows":
        result["platform"] = "windows"
    else:
        result["platform"] = sys_name

    if not Path(path).exists():
        result["message"] = f"Secrets file not found: {path}"
        logger.debug(result["message"])
        return result

    result["exists"] = True

    # Windows: no stat-based enforcement, just warn
    if result["platform"] == "windows":
        result["action"] = "warning"
        result["message"] = (
            f"Secrets file found at {path}. "
            "On Windows, please ensure this file is not shared or readable by "
            "other users. Right-click → Properties → Security to restrict access."
        )
        logger.warning(result["message"])
        return result

    # POSIX: check and optionally fix permissions
    try:
        st = stat_fn(str(path))
        mode = stat.S_IMODE(st.st_mode)
        result["mode_octal"] = oct(mode)

        if mode & _GROUP_OTHER_MASK:
            result["is_safe"] = False
            if enforce:
                try:
                    chmod_fn(str(path), _SAFE_MODE)
                    result["action"] = "fixed"
                    result["message"] = (
                        f"Secrets file {path} was {oct(mode)}, "
                        f"fixed to {oct(_SAFE_MODE)} (owner read/write only)."
                    )
                    logger.warning(result["message"])
                except OSError as exc:
                    result["action"] = "warning"
                    result["message"] = (
                        f"Secrets file {path} is {oct(mode)} (world-readable). "
                        f"Could not chmod: {exc}. "
                        "Please run: chmod 600 " + str(path)
                    )
                    logger.warning(result["message"])
            else:
                result["action"] = "warning"
                result["message"] = (
                    f"Secrets file {path} is {oct(mode)} — "
                    "group/other bits are set. Recommend: chmod 600 " + str(path)
                )
                logger.warning(result["message"])
        else:
            result["is_safe"] = True
            result["action"] = "ok"
            result["message"] = f"Secrets file {path} permissions are safe ({oct(mode)})."
            logger.debug(result["message"])

    except OSError as exc:
        result["action"] = "error"
        result["message"] = f"Could not stat secrets file {path}: {exc}"
        logger.error(result["message"])

    return result
