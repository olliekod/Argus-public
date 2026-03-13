"""
Shared test fixtures and environment stubs.

Stubs out packages that cannot be built in some CI environments
(``multitasking``, ``curl_cffi``, ``telegram``, etc.) so that
transitive importers can import cleanly.
"""

import sys
import types

# ── Stub multitasking ────────────────────────────────────────────────
if "multitasking" not in sys.modules:
    _mt = types.ModuleType("multitasking")
    _mt.__version__ = "0.0.11"
    _mt.task = lambda f: f  # no-op decorator
    sys.modules["multitasking"] = _mt

# ── Stub curl_cffi (yfinance >= 1.2.0 dependency) ───────────────────
if "curl_cffi" not in sys.modules:
    _cc = types.ModuleType("curl_cffi")
    _cc_req = types.ModuleType("curl_cffi.requests")
    _cc.requests = _cc_req
    sys.modules["curl_cffi"] = _cc
    sys.modules["curl_cffi.requests"] = _cc_req

# ── Stub yfinance ────────────────────────────────────────────────────
try:
    import yfinance  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _yf = types.ModuleType("yfinance")
    _yf.__version__ = "0.0.0"

    class _FakeTicker:
        def __init__(self, *a, **kw):
            pass

    _yf.Ticker = _FakeTicker
    sys.modules["yfinance"] = _yf

# ── Stub python-telegram-bot ─────────────────────────────────────────
# The telegram package requires cryptography which may have broken cffi
# bindings in some environments.  Stub the entire telegram tree so that
# src/alerts/telegram_bot.py can be imported during test collection.
try:
    from telegram import Bot  # noqa: F401
except Exception:
    # Base classes shared by all stubs
    class _TelegramError(Exception):
        pass

    class _NetworkError(_TelegramError):
        pass

    class _TimedOut(_TelegramError):
        pass

    class _Conflict(_TelegramError):
        pass

    class _Bot:
        def __init__(self, *a, **kw):
            pass

    class _Update:
        pass

    class _Application:
        @classmethod
        def builder(cls):
            return cls()
        def token(self, *a, **kw):
            return self
        def build(self):
            return self

    class _ContextTypes:
        DEFAULT_TYPE = None

    def _noop_handler(*a, **kw):
        pass

    # telegram (top-level)
    _tg = types.ModuleType("telegram")
    _tg.Bot = _Bot
    _tg.Update = _Update
    sys.modules["telegram"] = _tg

    # telegram.ext
    _tg_ext = types.ModuleType("telegram.ext")
    _tg_ext.Application = _Application
    _tg_ext.CommandHandler = _noop_handler
    _tg_ext.MessageHandler = _noop_handler
    _tg_ext.ContextTypes = _ContextTypes
    _tg_ext.filters = types.ModuleType("telegram.ext.filters")
    _tg.ext = _tg_ext
    sys.modules["telegram.ext"] = _tg_ext
    sys.modules["telegram.ext.filters"] = _tg_ext.filters

    # telegram.error
    _tg_err = types.ModuleType("telegram.error")
    _tg_err.TelegramError = _TelegramError
    _tg_err.NetworkError = _NetworkError
    _tg_err.TimedOut = _TimedOut
    _tg_err.Conflict = _Conflict
    _tg.error = _tg_err
    sys.modules["telegram.error"] = _tg_err
