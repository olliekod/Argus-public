"""
Kalshi REST + WebSocket authentication.

Signing scheme
--------------
For every signed request Kalshi requires:

    message = str(timestamp_ms) + METHOD + PATH

where PATH is the **full** path including the API prefix (e.g. ``/trade-api/v2/portfolio/orders``),
without query string.

The message is signed with **RSA-PSS / SHA-256**, then base64-encoded.

Three headers are attached:
    KALSHI-ACCESS-KEY         — the API key ID
    KALSHI-ACCESS-TIMESTAMP   — the millisecond timestamp used in the message
    KALSHI-ACCESS-SIGNATURE   — the base64-encoded signature

WebSocket authentication
------------------------
The same three headers are included during the WebSocket HTTP upgrade
handshake.  The signed path defaults to the WS URL path
(``/trade-api/ws/v2``), but can be overridden via ``ws_signing_path``
in config, because Kalshi's signing validator MAY expect a different
path than what appears in the URL (this is exchange-specific and has
changed in the past).  The method used for signing is ``GET``.

**Verification checklist before enabling ws_trading_enabled=True:**
1. Confirm the WS handshake succeeds against production (not just demo).
2. Confirm which path Kalshi's signing validator expects for WS.
3. Confirm that private channels (fill, user_orders) return data.

Clock drift / NTP
-----------------
Kalshi validates that the timestamp is close to server time.

**Mandatory**: the host MUST run NTP (chrony, systemd-timesyncd, etc.).

**Optional**: if ``enable_clock_offset_calibration`` is set in config, we
call a lightweight Kalshi endpoint, read the ``Date`` header from the
response, and compute an offset.  This offset is then added to all future
timestamps.  If the calibration endpoint is unreachable, we log a warning
and proceed with zero offset — NTP is then the sole guard.

**Offset bounding**: Date headers can come from CDN edge proxies that are
not aligned with the signing validator.  We reject offsets whose absolute
value exceeds ``max_clock_offset_ms`` (default 5 s) and fall back to 0.
A bad offset is worse than no offset.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes

from .logging_utils import ComponentLogger

log = ComponentLogger("auth")


def load_private_key(pem_path: str, password: Optional[bytes] = None) -> PrivateKeyTypes:
    """Load an RSA private key from a PEM file.

    Parameters
    ----------
    pem_path:
        Filesystem path to the PEM-encoded private key.
    password:
        Optional passphrase for encrypted keys.

    Returns
    -------
    The loaded private key object.
    """
    data = Path(pem_path).read_bytes()
    key = serialization.load_pem_private_key(data, password=password)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(f"Expected RSA private key, got {type(key).__name__}")
    return key


def sign_message(private_key: PrivateKeyTypes, message: str) -> str:
    """Sign *message* with RSA-PSS / SHA-256 and return a base64 string."""
    sig_bytes = private_key.sign(  # type: ignore[union-attr]
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig_bytes).decode("ascii")


def build_headers(
    key_id: str,
    private_key: PrivateKeyTypes,
    method: str,
    full_url_or_path: str,
    offset_ms: int = 0,
) -> Dict[str, str]:
    """Return the three Kalshi auth headers for a single request.

    Parameters
    ----------
    key_id:
        ``KALSHI-ACCESS-KEY`` value.
    private_key:
        Loaded RSA private key.
    method:
        HTTP method (``GET``, ``POST``, ``DELETE``, …) — upper-cased internally.
    full_url_or_path:
        Either a full URL or just the path component.  Any query string is
        stripped before signing.
    offset_ms:
        Milliseconds to add to ``time.time_ns() // 1_000_000`` to compensate
        for clock drift.  Obtained via :func:`calibrate_clock_offset`.
    """
    # Strip query string — Kalshi signs only the path.
    parsed = urlparse(full_url_or_path)
    path = parsed.path  # e.g. "/trade-api/v2/markets"

    ts_ms = int(time.time() * 1000) + offset_ms
    msg = f"{ts_ms}{method.upper()}{path}"
    signature = sign_message(private_key, msg)

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


# ---------------------------------------------------------------------------
#  Optional clock offset calibration
# ---------------------------------------------------------------------------

async def calibrate_clock_offset(
    session: "aiohttp.ClientSession",  # noqa: F821 — forward ref
    base_url: str,
    max_offset_ms: int = 5000,
) -> int:
    """Estimate ``server_ms - local_ms`` by calling a public Kalshi endpoint.

    We use ``GET /trade-api/v2/exchange/status`` which is unauthenticated and
    returns a ``Date`` header.  If the endpoint changes or is unavailable, we
    return 0 and log a warning.

    Parameters
    ----------
    max_offset_ms:
        Reject offsets whose absolute value exceeds this bound.  Date
        headers can be served by CDN edge proxies that are not aligned
        with the signing validator; large offsets are likely noise.

    Returns
    -------
    Offset in milliseconds to add to local timestamps.
    """
    import email.utils
    from datetime import datetime, timezone

    url = f"{base_url.rstrip('/')}/exchange/status"
    try:
        local_before_ms = int(time.time() * 1000)
        async with session.get(url, timeout=__import__("aiohttp").ClientTimeout(total=5)) as resp:
            local_after_ms = int(time.time() * 1000)
            date_str = resp.headers.get("Date")
            if not date_str:
                log.warning(
                    "No Date header in calibration response; using offset=0. "
                    "NTP sync is mandatory."
                )
                return 0

            # Date header has only second resolution — inherently coarse.
            server_dt = email.utils.parsedate_to_datetime(date_str)
            server_ms = int(server_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
            local_mid_ms = (local_before_ms + local_after_ms) // 2
            rtt_ms = local_after_ms - local_before_ms
            offset = server_ms - local_mid_ms

            trusted = abs(offset) <= max_offset_ms
            log.info(
                "Clock offset calibrated",
                data={
                    "offset_ms": offset,
                    "rtt_ms": rtt_ms,
                    "server_date": date_str,
                    "trusted": trusted,
                },
            )

            if not trusted:
                log.warning(
                    f"Clock offset {offset}ms exceeds bound ±{max_offset_ms}ms; "
                    f"ignoring (likely CDN proxy). NTP sync is mandatory."
                )
                return 0

            return offset
    except Exception as exc:
        log.warning(
            f"Clock offset calibration failed: {exc}; NTP sync is mandatory"
        )
        return 0
