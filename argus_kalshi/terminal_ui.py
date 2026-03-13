"""
Argus Vision - Terminal UI v5 (NGE)
====================================
"""

import asyncio
import json
import os
import sys
import time
import shutil
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Crash log for separate UI process (same path as ui_client, cwd-based)
def _ui_crash_log_path() -> str:
    return os.path.join(os.getcwd(), "logs", "argus_ui_crash.log")


def _write_ui_crash(kind: str, e: Exception) -> None:
    try:
        path = _ui_crash_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {kind} at {datetime.now(timezone.utc).isoformat()} ---\n")
            traceback.print_exc(file=f)
    except Exception:
        pass


def _ui_build_stamp() -> str:
    """Stable per-build stamp from this module's file mtime."""
    try:
        ts = os.path.getmtime(__file__)
        return datetime.fromtimestamp(ts).strftime("%Y%m%d-%H%M")
    except Exception:
        return "unknown"

from .bus import Bus
from .logging_utils import ComponentLogger
from .models import (
    AccountBalance,
    BtcMidPrice,
    FairProbability,
    FillEvent,
    KalshiRtt,
    MarketMetadata,
    OrderUpdate,
    OrderbookState,
    SettlementOutcome,
    TerminalEvent,
    TradeSignal,
)

log = ComponentLogger("terminal_ui")

try:
    import colorama
    colorama.init()
except ImportError:
    pass

# ── ANSI ──────────────────────────────────────────────────────────────
CYAN    = "\033[38;5;51m"
ORANGE  = "\033[38;5;208m"
AMBER   = "\033[38;5;214m"
WHITE   = "\033[38;5;253m"
GREEN   = "\033[38;5;42m"
RED     = "\033[38;5;196m"
YELLOW  = "\033[38;5;226m"
# Re-map gray tones to brighter, more legible hues.
# GRAY becomes soft white, DKGRAY becomes a muted cyan accent.
GRAY    = "\033[38;5;253m"
DKGRAY  = "\033[38;5;110m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
DIM     = "\033[2m"
CLR     = "\033[K"          # clear to end of line

# ── Animation constants ────────────────────────────────────────────────
_SPINNERS = ["◈", "◇", "◆", "◉"]
_ARROWS   = ["▶▶", "▷▷"]
_WAVE     = "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂"
_SPARK    = "▁▂▃▄▅▆▇█"
_ROTATORS = ["|", "/", "-", "\\", "|", "/", "-", "\\"]
_LIGHTS   = ["●○○", "○●○", "○○●", "○●○"]

# ── NGE-style panel animations (data-driven; each panel uses real metrics) ─
_SCAN_SWEEP_FRAMES = [
    ["│░       │", "│░       │", "│░       │", "│░       │", "│░       │"],
    ["│ ░      │", "│ ░      │", "│ ░      │", "│ ░      │", "│ ░      │"],
    ["│  ░     │", "│  ░     │", "│  ░     │", "│  ░     │", "│  ░     │"],
    ["│   ░    │", "│   ░    │", "│   ░    │", "│   ░    │", "│   ░    │"],
    ["│    ░   │", "│    ░   │", "│    ░   │", "│    ░   │", "│    ░   │"],
    ["│     ░  │", "│     ░  │", "│     ░  │", "│     ░  │", "│     ░  │"],
    ["│      ░ │", "│      ░ │", "│      ░ │", "│      ░ │", "│      ░ │"],
]
_RADAR_FRAMES = [
    "[██████      ]", "[ ██████     ]", "[  ██████    ]", "[   ██████   ]",
    "[    ██████  ]", "[     ██████ ]", "[      ██████]",
]
# Vertical bar levels 0–8 (8 rows); first block = fill up, second = drain
_BAR_LEVELS = (" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
_SCOPE_WAVE = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
# 3-column grid so KXBTC / KXETH / KXSOL line up with the three prongs
_GRID_FRAMES = [
    ["┌─────┬─────┬─────┐", "│▒▒▒▒▒│     │     │", "├─────┼─────┼─────┤", "│     │▒▒▒▒▒│     │", "└─────┴─────┴─────┘"],
    ["┌─────┬─────┬─────┐", "│     │▒▒▒▒▒│     │", "├─────┼─────┼─────┤", "│     │     │▒▒▒▒▒│", "└─────┴─────┴─────┘"],
    ["┌─────┬─────┬─────┐", "│     │     │▒▒▒▒▒│", "├─────┼─────┼─────┤", "│▒▒▒▒▒│     │     │", "└─────┴─────┴─────┘"],
]
_FRAME_ACTIVE = ["ACTIVE   ", "ACTIVE.  ", "ACTIVE.. ", "ACTIVE..."]
_BACKGROUND_FRAMES = [
    ["░░░░░░░░░░░░░░░░░░", "                   ", "░░░░░░░░░░░░░░░░░░"],
    ["                   ", "░░░░░░░░░░░░░░░░░░", "                   "],
]

# ── Thresholds ────────────────────────────────────────────────────────
EDGE_THRESHOLD_CENTS = 1
SPREAD_MAX_CENTS     = 5
DEFAULT_W = 90   # fallback width
DEFAULT_REMOTE_ROWS = 80
UI_TARGET_FPS = 16
UI_MAX_RENDER_HZ = 16.0  # allow 16 fps when inline so animations stay smooth
UI_EVENT_BUDGET_SEC = 0.02
MIN_SETTLED_TRADES_FOR_ALPHA = 5
UI_MIN_LAYOUT_WIDTH = 126  # keep top sections aligned with leaderboard width
# Cache TTLs to keep UI smooth with many bots (e.g. 468)
LEADERBOARD_CACHE_TTL = 0.4   # Recompute top-20 alpha at most every 0.4s
TERMINAL_SIZE_CACHE_TTL = 1.0  # Refresh terminal size at most every 1s
# In separate-process UI, cap iteration to avoid first-render overload / Windows console crash
REMOTE_MODE_MAX_STATES = 1000

DEFAULT_DISPLAY_ASSETS = ("BTC", "ETH")
UI_HOLD_MIN_EDGE_CENTS = 7
UI_HOLD_MIN_ENTRY_CENTS = 35
UI_HOLD_MAX_ENTRY_CENTS = 75
UI_NO_AVOID_ABOVE_CENTS = 75
UI_SCALP_MIN_EDGE_CENTS = 16
UI_SCALP_MIN_ENTRY_CENTS = 30
UI_SCALP_MAX_ENTRY_CENTS = 65
UI_SCALP_MAX_SPREAD_CENTS = 2
PROMOTED_METADATA_CACHE_TTL = 5.0
PROMOTED_JSON_PATH = Path("config/kalshi_promoted_bot.json")
LIFETIME_JSON_PATH = Path("config/kalshi_lifetime_performance.json")


# Scoring functions and scenario projections are now in simulation.py.
# Import and re-export for backward compatibility.
from .simulation import (
    calculate_robustness_score,
    calculate_alpha_score,
    project_execution_scenarios,
    SCALP_BEST_RELIEF_USD as SCALP_BEST_RELIEF_USD_PER_CONTRACT,
    HOLD_BEST_RELIEF_USD as HOLD_BEST_RELIEF_USD_PER_CONTRACT,
    SCALP_STRESS_DRAG_USD as SCALP_STRESS_DRAG_USD_PER_CONTRACT,
    HOLD_STRESS_DRAG_USD as HOLD_STRESS_DRAG_USD_PER_CONTRACT,
    MIN_SETTLED_TRADES_FOR_SCORE as MIN_SETTLED_TRADES_FOR_ALPHA,
)


# ======================================================================
#  Helpers
# ======================================================================

def _sparkline(prices: deque, width: int = 10) -> str:
    """Convert a price history deque to a unicode sparkline string."""
    data = list(prices)
    if len(data) < 2:
        return "─" * width
    lo, hi = min(data), max(data)
    rng = hi - lo
    if rng == 0:
        return "─" * width
    if len(data) > width:
        data = data[-width:]
    return "".join(
        _SPARK[min(7, int((v - lo) / rng * 7.99))]
        for v in data
    ).rjust(width, " ")


def _wave_segment(frame: int, width: int, phase: int = 0) -> str:
    """Return a short looping wave segment for section headers / accents."""
    if width <= 0:
        return ""
    off = (frame + phase) % len(_WAVE)
    return "".join(_WAVE[(i + off) % len(_WAVE)] for i in range(width))


def _header_lights(frame: int) -> str:
    """Shared accent for all major section headers (NGE status lights)."""
    return _LIGHTS[frame % len(_LIGHTS)]


def _markets_sep_line(frame: int, width: int, phase: int = 0, speed_div: int = 4) -> str:
    """Animated dotted separator for MARKETS: a single circle flows along the line.
    speed_div: larger = slower (e.g. 4, 5, 6 for BTC, ETH, SOL).
    """
    if width <= 0:
        return ""
    pos = (frame // speed_div + phase) % width
    return "".join("●" if j == pos else "·" for j in range(width))


def _prob_bar(p_yes: float, width: int = 10) -> str:
    """Render a filled probability bar using block characters."""
    filled = round(p_yes * width)
    return f"{'▓' * filled}{'░' * (width - filled)}"


def _market_label(ticker: str, source: str = "") -> str:
    """Convert a raw Kalshi ticker to a short human-readable label.

    Format: ``ASSET WINDOW TYPE``

    - ASSET  = BTC / ETH / SOL
    - WINDOW = 15m / 60m / Range / Daily
    - TYPE   = E (hold to expiry / settlement strategy)
              S (mispricing scalp — early exit)
              A (pair arbitrage sleeve)

    Examples::

        KXBTC15M-26MAR0217-B65000              → "BTC 15m E"
        KXBTCH-26MAR0617-B66500                → "BTC 60m E"
        KXBTC-26MAR0217-B65000                 → "BTC Range E"
        KXETH15M-... source=mispricing_scalp   → "ETH 15m S"
        KXBTC15M-... source=pair_arb           → "BTC 15m A"
    """
    series = ticker.upper().split("-")[0] if "-" in ticker else ticker.upper()

    if "BTC" in series:
        asset = "BTC"
    elif "ETH" in series:
        asset = "ETH"
    elif "SOL" in series:
        asset = "SOL"
    else:
        asset = ticker[:3].upper()

    if series.endswith("15M"):
        window = "15m"
    elif series.endswith("H"):
        window = "60m"
    elif series.endswith("D"):
        window = "Daily"
    elif series in ("KXBTC", "KXETH", "KXSOL"):
        window = "Range"
    else:
        window = "?"

    if source == "mispricing_scalp":
        suffix = "S"
    elif source == "pair_arb":
        suffix = "A"
    else:
        suffix = "E"
    return f"{asset} {window} {suffix}"


def _source_bucket(source: str) -> str:
    src = (source or "").strip()
    if src == "mispricing_scalp":
        return "scalp"
    if src == "pair_arb":
        return "arb"
    return "expiry"


def _ui_hold_family_allowed(asset: str, window_label: str) -> bool:
    if asset == "SOL":
        return False
    if asset == "BTC":
        return window_label in {"15min", "60min", "Range"}
    if asset == "ETH":
        return window_label in {"15min", "60min"}
    return False


def _ui_scalp_family_allowed(asset: str, window_label: str) -> bool:
    if asset == "BTC":
        return window_label in {"15min", "60min"}
    if asset == "ETH":
        return window_label in {"15min", "60min"}
    return False


# ======================================================================
#  TickerState — per-market state
# ======================================================================

class TickerState:
    __slots__ = (
        "ticker", "asset", "p_yes", "p_yes_hist",
        "yes_ask", "no_ask", "yes_bid", "no_bid",
        "ob_valid", "ob_had_valid", "exp_ts", "window_min", "is_range",
        "strike", "strike_floor", "strike_cap", "is_directional",
    )

    def __init__(self, ticker: str, meta: MarketMetadata):
        self.ticker      = ticker
        self.asset       = meta.asset
        self.p_yes: float          = 0.5
        self.p_yes_hist: deque     = deque(maxlen=20)
        self.yes_ask: int          = 0
        self.no_ask: int           = 0
        self.yes_bid: int          = 0
        self.no_bid: int           = 0
        self.ob_valid: bool        = False
        self.ob_had_valid: bool    = False   # True once any valid snapshot received
        self.exp_ts: float         = 0.0
        self.window_min: int       = meta.window_minutes
        self.is_range: bool        = meta.is_range
        self.strike: float         = meta.strike_price
        self.strike_floor: Optional[float] = meta.strike_floor
        self.strike_cap: Optional[float]   = meta.strike_cap
        self.is_directional: bool          = meta.is_directional
        try:
            self.exp_ts = datetime.fromisoformat(
                meta.settlement_time_iso.replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            pass

    # ── derived properties ────────────────────────────────────────────

    @property
    def edge_yes(self) -> int:
        return round(self.p_yes * 100) - self.yes_ask if self.ob_valid else 0

    @property
    def edge_no(self) -> int:
        return round((1.0 - self.p_yes) * 100) - self.no_ask if self.ob_valid else 0

    @property
    def best_edge(self) -> int:
        return max(self.edge_yes, self.edge_no)

    @property
    def best_side(self) -> str:
        return "YES" if self.edge_yes >= self.edge_no else "NO"

    @property
    def best_ask(self) -> int:
        return self.yes_ask if self.edge_yes >= self.edge_no else self.no_ask

    @property
    def spread(self) -> int:
        if not self.ob_valid:
            return 0
        return max(0, (self.yes_ask - self.yes_bid) + (self.no_ask - self.no_bid))

    @property
    def window_label(self) -> str:
        if self.is_range:
            return "Range"
        return f"{self.window_min}min"

    @property
    def strike_display(self) -> str:
        if self.is_directional:
            if self.strike > 0:
                return f"${self.strike:,.0f}"  # BRTI reference from floor_strike
            return "↑↓ DIR"
        if self.is_range:
            lo = self.strike_floor
            hi = self.strike_cap
            if lo is not None and hi is not None:
                return f"${lo:,.0f}-${hi:,.0f}"
            return "---"
        if self.strike <= 0:
            return "---"
        return f"${self.strike:,.0f}"

    @property
    def time_remaining_raw(self) -> float:
        if self.exp_ts <= 0: return 0
        return self.exp_ts - time.time()

    @property
    def time_remaining(self) -> str:
        rem = int(self.time_remaining_raw)
        if rem <= 0:
            return "EXPRD"
        # Kalshi opens 15m/60m contracts before their window starts — cap the
        # display so they never show more than their window duration.
        # Range contracts have no fixed time window (window_min is 0 from the
        # API), so they are exempt and always show their actual settlement countdown.
        if not self.is_range:
            rem = min(rem, self.window_min * 60)
        d, r = divmod(rem, 86400)
        h, r = divmod(r, 3600)
        m, s = divmod(r, 60)
        if d > 0:
            return f"{d}d{h}h"
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m:02d}:{s:02d}"

    def is_actionable(self) -> bool:
        return (
            self.ob_valid
            and self.best_edge >= EDGE_THRESHOLD_CENTS
            and self.spread <= SPREAD_MAX_CENTS
        )


# ======================================================================
#  TerminalVisualizer
# ======================================================================

class TerminalVisualizer:
    """
    NGE-inspired terminal dashboard rendering at ~20 fps.

    No alternate screen buffer is used, so copy-paste works normally.
    The screen is cleared once at start, then the cursor homes each frame.
    """

    def __init__(
        self,
        bus: Optional[Bus] = None,
        metadata: Dict[str, MarketMetadata] = None,
        dry_run: bool = True,
        primary_bot_id: Optional[str] = None,
        leaderboard_only: bool = False,
        unique_config_count: Optional[int] = None,
        initial_kalshi_rtt_ms: Optional[float] = None,
        remote_snapshot_queue: Optional[asyncio.Queue] = None,
        checkpoint_cb: Optional[Callable[[str], None]] = None,
    ):
        self._bus      = bus
        self._remote_snapshot_queue = remote_snapshot_queue
        self._checkpoint_cb = checkpoint_cb
        self._remote_minimal_render_done = False
        self._metadata: Dict[str, MarketMetadata] = metadata or {}
        self._dry_run  = dry_run
        self._running  = False
        self._task: Optional[asyncio.Task] = None
        _pid = (primary_bot_id or "").strip()
        self._primary_bot_id = _pid if _pid and _pid.lower() != "default" else None
        self._leaderboard_only = leaderboard_only
        self._unique_config_count = unique_config_count
        self._ui_build_stamp = _ui_build_stamp()

        # Prices + sparkline history (sampled every 5 frames ≈ 0.25s)
        self._prices: Dict[str, float] = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
        self._price_hist: Dict[str, deque] = {
            "BTC": deque(maxlen=20),
            "ETH": deque(maxlen=20),
            "SOL": deque(maxlen=20),
        }

        # Market states (only for subscribed near-money markets)
        self._states: Dict[str, TickerState] = {}

        # Performance tracking
        self._session_pnl: float = 0.0
        self._alltime_pnl: float = 0.0
        self._wins:   int = 0
        self._losses: int = 0

        # Order / fill tracking
        self._recent_fills: deque = deque(maxlen=500)
        self._total_fills: int = 0
        self._open_orders: int = 0

        # Settlement history
        self._history: deque = deque(maxlen=500)

        # Account balance (polled from Kalshi API every 60s)
        self._balance_usd: Optional[float] = None
        self._initial_balance_usd: Optional[float] = None  # first snapshot = "principal"

        # Kalshi RTT (ms); REST and/or WebSocket; set at startup and updated via bus
        self._kalshi_rtt_ms: Optional[float] = initial_kalshi_rtt_ms
        self._kalshi_rtt_source: Optional[str] = None  # "rest" | "ws" for display

        # Extended stats
        self._win_pnl_total: float  = 0.0   # sum of all winning trade PnL
        self._loss_pnl_total: float = 0.0   # sum of all losing trade PnL
        self._win_streak:  int = 0
        self._loss_streak: int = 0
        self._best_win:    float = 0.0
        self._worst_loss:  float = 0.0
        self._total_contracts: int = 0      # cumulative contracts filled

        # Leaderboard Tracking
        # Maps bot_id to a dict of { pnl, wins, losses, balance_offset }
        self._bot_stats: Dict[str, Dict[str, Any]] = {}
        
        # Health Check
        self._ws_connected = False

        # Telemetry
        self._start_time = time.time()
        self._runtime_start_time = self._start_time
        self._frame = 0
        self._counts = {"mkt": 0, "tick": 0, "prob": 0, "ob": 0}
        # Active open orders by market bucket (legacy name kept for compatibility).
        self._market_fill_counts: Dict[str, int] = {}
        self._open_order_bucket_by_oid: Dict[str, str] = {}
        self._last_render_ts: float = 0.0

        # Performance caching
        self._bot_leaderboard_cache: List[tuple[str, Dict[str, Any], float]] = []
        self._last_leaderboard_update: float = 0.0
        self._best_markets_cache: Dict[str, Dict[str, Optional[TickerState]]] = {}
        self._last_best_markets_update: float = 0.0
        self._terminal_size: tuple = (DEFAULT_W, 40)
        self._terminal_size_ts: float = 0.0

        # NGE panels: per-panel frame offset so they animate asynchronously
        self._panel_scan = 0
        self._panel_radar = 0
        self._panel_bars = 0
        self._panel_scope = 0
        self._panel_grid = 0
        self._panel_frame = 0
        self._panel_bg = 0
        self._panel_cascade = 0
        # Population/evolution telemetry (from IPC snapshot["population"])
        self._population_enabled: bool = False
        self._population_generation: int = 0
        self._population_epoch_count: int = 0
        self._population_reseed_total: int = 0
        self._population_reseed_epoch: int = 0
        self._population_reseed_drawdown: int = 0
        self._population_last_epoch_ts: float = 0.0
        self._population_last_reseed_ts: float = 0.0
        self._population_epoch_minutes: float = 0.0
        # Streaming data for cascade: last N ticker/price snippets (real data)
        self._cascade_buffer: deque = deque(maxlen=12)
        # Promoted bot lifetime metadata (file-backed, cached).
        self._promoted_meta: Dict[str, Any] = {}
        self._promoted_meta_load_ts: float = 0.0
        self._lifetime_max_fills: int = 0
        self._promoted_session_peak_pnl: float = 0.0

    # ── public interface ──────────────────────────────────────────────

    def set_initial_stats(
        self,
        total_pnl: float,
        wins: int,
        total: int,
        first_entry_ts: float = 0.0,
        seed_balance_usd: Optional[float] = None,
    ):
        """Load all-time stats from JSONL on startup.

        first_entry_ts — Unix timestamp of the first JSONL record.  When
        provided, the visualiser treats that moment as the bot's origin so
        uptime and rate stats span the full history, not just this session.

        seed_balance_usd — Pre-populate the balance display immediately so
        "Awaiting balance update..." never shows on startup.  The periodic
        balance poller will overwrite this with the live API value shortly.
        """
        self._alltime_pnl = total_pnl
        # session_pnl starts at 0 — tracks only what was earned THIS session
        self._wins   = wins
        self._losses = max(0, total - wins)
        if first_entry_ts > 0:
            self._start_time = first_entry_ts
        if seed_balance_usd is not None:
            self._balance_usd = seed_balance_usd
            self._initial_balance_usd = seed_balance_usd
        self._promoted_session_peak_pnl = 0.0

    def seed_bot_stats(self, bot_stats: Dict[str, Dict[str, Any]]) -> None:
        """Seed dwarf ranking from paper_trades.jsonl so Net PnL is correct after restart."""
        self._bot_stats = {k: dict(v) for k, v in bot_stats.items()}

    def ensure_bot_stats_entries(self, bot_ids: List[str]) -> None:
        """Pre-seed empty stats for all known bot_ids so the leaderboard shows dwarf names before they trade."""
        empty = {
            "pnl": 0.0, "pnl_e": 0.0, "pnl_s": 0.0, "pnl_a": 0.0,
            "gross_pnl": 0.0, "fees_usd": 0.0,
            "qty_e_contracts": 0.0, "qty_s_contracts": 0.0, "qty_a_contracts": 0.0,
            "wins": 0, "losses": 0, "fills": 0, "orders": 0,
            "fills_e": 0, "fills_s": 0, "fills_a": 0,
            "trade_count": 0, "gross_profit": 0.0, "gross_loss": 0.0,
            "peak_pnl": 0.0, "max_drawdown": 0.0, "last_active": 0.0,
            "tail_loss_10pct": 0.0,
            "generation": 0, "run_id": "", "parent_run_id": "",
            "family_pnl": {},
            "start_equity": 5000.0, "scenario": "base",
            "equity": 5000.0, "max_drawdown_pct": 0.0,
        }
        for bid in bot_ids:
            if bid and bid not in self._bot_stats:
                self._bot_stats[bid] = dict(empty)

    async def start(self) -> None:
        self._running = True
        # Clear screen once; use cursor-home per frame (no alternate buffer)
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        self._running = False
        # Restore cursor, reset colors, add newline for clean exit
        sys.stdout.write("\033[?25h" + RESET + "\n")
        sys.stdout.flush()
        if self._task:
            self._task.cancel()

    async def update_markets(self, added: Dict[str, MarketMetadata], removed: List[str]) -> None:
        """Dynamically update tracked markets during a session."""
        for ticker in removed:
            self._metadata.pop(ticker, None)
            self._states.pop(ticker, None)
        for ticker, meta in added.items():
            self._metadata[ticker] = meta
            self._ensure_state(meta)

    def update_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Update UI state from an IPC snapshot (used when UI runs in separate process)."""
        if not snapshot:
            return
        # Apply primary_bot_id from backend so promoted slot shows correct bot
        pid = snapshot.get("primary_bot_id")
        if pid is not None:
            _pid = (pid or "").strip()
            self._primary_bot_id = _pid if _pid and _pid.lower() != "default" else None
        self._prices = dict(snapshot.get("prices", self._prices))
        self._balance_usd = snapshot.get("balance_usd")
        if snapshot.get("initial_balance_usd") is not None:
            self._initial_balance_usd = snapshot["initial_balance_usd"]
        self._kalshi_rtt_ms = snapshot.get("kalshi_rtt_ms")
        self._kalshi_rtt_source = snapshot.get("kalshi_rtt_source")
        self._ws_connected = bool(snapshot.get("ws_connected", False))
        self._session_pnl = float(snapshot.get("session_pnl", 0))
        self._alltime_pnl = float(snapshot.get("alltime_pnl", 0))
        self._promoted_session_peak_pnl = float(
            snapshot.get("primary_session_peak_pnl", max(self._promoted_session_peak_pnl, self._session_pnl))
        )
        self._wins = int(snapshot.get("wins", 0))
        self._losses = int(snapshot.get("losses", 0))
        self._win_pnl_total = float(snapshot.get("win_pnl_total", 0))
        self._loss_pnl_total = float(snapshot.get("loss_pnl_total", 0))
        self._win_streak = int(snapshot.get("win_streak", 0))
        self._loss_streak = int(snapshot.get("loss_streak", 0))
        self._best_win = float(snapshot.get("best_win", 0))
        self._worst_loss = float(snapshot.get("worst_loss", 0))
        self._total_contracts = int(snapshot.get("total_contracts", 0))
        self._open_orders = int(snapshot.get("open_orders", 0))
        self._market_fill_counts = dict(snapshot.get("market_fill_counts") or {})
        population = dict(snapshot.get("population") or {})
        self._population_enabled = bool(population.get("enabled", False))
        self._population_generation = int(population.get("generation", 0) or 0)
        self._population_epoch_count = int(population.get("epoch_count", 0) or 0)
        self._population_reseed_total = int(population.get("reseed_total", 0) or 0)
        self._population_reseed_epoch = int(population.get("reseed_epoch", 0) or 0)
        self._population_reseed_drawdown = int(population.get("reseed_drawdown", 0) or 0)
        self._population_last_epoch_ts = float(population.get("last_epoch_ts", 0.0) or 0.0)
        self._population_last_reseed_ts = float(population.get("last_reseed_ts", 0.0) or 0.0)
        self._population_epoch_minutes = float(population.get("epoch_minutes", 0.0) or 0.0)
        bot_items = list((snapshot.get("bot_stats") or {}).items())
        self._bot_stats = {k: dict(v) for k, v in bot_items}
        metadata_items = list((snapshot.get("metadata") or {}).items())
        if self._remote_snapshot_queue is not None and len(metadata_items) > REMOTE_MODE_MAX_STATES:
            metadata_items = metadata_items[:REMOTE_MODE_MAX_STATES]
        for ticker, meta_dict in metadata_items:
            try:
                meta = MarketMetadata(**meta_dict)
                self._metadata[ticker] = meta
                if ticker not in self._states:
                    self._states[ticker] = TickerState(ticker, meta)
            except (TypeError, KeyError):
                continue
        states_items = list((snapshot.get("states") or {}).items())
        if self._remote_snapshot_queue is not None and len(states_items) > REMOTE_MODE_MAX_STATES:
            states_items = states_items[:REMOTE_MODE_MAX_STATES]
        for ticker, state_dict in states_items:
            if ticker not in self._states:
                meta_dict = (snapshot.get("metadata") or {}).get(ticker)
                if meta_dict:
                    try:
                        meta = MarketMetadata(**meta_dict)
                        self._metadata[ticker] = meta
                        self._states[ticker] = TickerState(ticker, meta)
                    except (TypeError, KeyError):
                        continue
            st = self._states.get(ticker)
            if st is not None:
                st.p_yes = state_dict.get("p_yes", 0.5)
                hist = state_dict.get("p_yes_hist") or []
                st.p_yes_hist = deque(hist[-20:], maxlen=20)
                st.yes_ask = int(state_dict.get("yes_ask", 0))
                st.no_ask = int(state_dict.get("no_ask", 0))
                st.yes_bid = int(state_dict.get("yes_bid", 0))
                st.no_bid = int(state_dict.get("no_bid", 0))
                st.ob_valid = bool(state_dict.get("ob_valid", False))
                st.ob_had_valid = bool(state_dict.get("ob_had_valid", False))
        self._recent_fills = deque(snapshot.get("recent_fills") or [], maxlen=500)
        self._history = deque(snapshot.get("history") or [], maxlen=500)
        # Telemetry for header (Mkts, Probs, OB) — derived from snapshot in remote mode.
        states = snapshot.get("states") or {}
        self._counts["mkt"] = len(states)
        self._counts["prob"] = sum(1 for s in states.values() if s.get("p_yes", 0.5) != 0.5)
        self._counts["ob"] = sum(1 for s in states.values() if s.get("ob_valid"))
        self._counts["tick"] = self._counts["ob"]  # no tick count in snapshot; use ob as proxy
        if self._frame % 5 == 0:
            for a in ["BTC", "ETH", "SOL"]:
                if self._prices.get(a, 0) > 0:
                    self._price_hist[a].append(self._prices[a])

    def render_frame(self) -> None:
        """Force one render (used by remote UI process after update_from_snapshot)."""
        # Remote mode: first time we have data, do a minimal one-line write to avoid Windows console crash on first big output.
        if self._remote_snapshot_queue is not None and not getattr(self, "_remote_minimal_render_done", False):
            if self._states or self._bot_stats:
                try:
                    sys.stdout.write("\033[H\n  Loading...\n")
                    sys.stdout.flush()
                except Exception:
                    pass
                self._remote_minimal_render_done = True
                self._last_render_ts = time.time()
                return
        try:
            self._render()
        except Exception as e:
            _write_ui_crash("render_frame", e)
            raise
        self._last_render_ts = time.time()

    # ── state management ──────────────────────────────────────────────

    def _ensure_state(self, m: MarketMetadata) -> None:
        if m.market_ticker not in self._states:
            self._states[m.market_ticker] = TickerState(m.market_ticker, m)

    def _market_bucket_key(self, ticker: str) -> Optional[str]:
        # Per-ticker open-order attribution so MARKETS.Ord reflects
        # the exact contract row, not an asset/window aggregate bucket.
        return ticker or None

    def _apply_order_bucket_update(self, ou: OrderUpdate) -> None:
        order_id = (ou.order_id or ou.client_order_id or "").strip()
        status = (ou.status or "").strip().lower()
        if not order_id or not status:
            return

        if status == "placed":
            bucket = self._market_bucket_key(ou.market_ticker)
            if not bucket:
                return
            if order_id in self._open_order_bucket_by_oid:
                return
            self._open_order_bucket_by_oid[order_id] = bucket
            self._market_fill_counts[bucket] = self._market_fill_counts.get(bucket, 0) + 1
            return

        if status in {"filled", "partial_fill", "cancelled", "canceled", "error"}:
            bucket = self._open_order_bucket_by_oid.pop(order_id, None)
            if not bucket:
                return
            cur = self._market_fill_counts.get(bucket, 0)
            if cur <= 1:
                self._market_fill_counts.pop(bucket, None)
            else:
                self._market_fill_counts[bucket] = cur - 1

    def _empty_bot_stats(self) -> Dict[str, Any]:
        return {
            "pnl": 0.0,
            "pnl_e": 0.0,  # expiry / hold-to-settlement PnL
            "pnl_s": 0.0,  # scalp / mispricing_scalp PnL
            "pnl_a": 0.0,  # pair-arb PnL
            "gross_pnl": 0.0,
            "fees_usd": 0.0,
            "qty_e_contracts": 0.0,
            "qty_s_contracts": 0.0,
            "qty_a_contracts": 0.0,
            "wins": 0,
            "losses": 0,
            "fills": 0,
            "fills_e": 0,
            "fills_s": 0,
            "fills_a": 0,
            "orders": 0,
            "trade_count": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "peak_pnl": 0.0,
            "max_drawdown": 0.0,
            "last_active": time.time(),
            "tail_loss_10pct": 0.0,
            "generation": 0,
            "run_id": "",
            "parent_run_id": "",
            "family_pnl": {},
            "start_equity": 5000.0,
            "scenario": "base",
            "equity": 5000.0,
            "max_drawdown_pct": 0.0,
        }

    @staticmethod
    def _event_bot_id(event: Any) -> Optional[str]:
        raw = getattr(event, "bot_id", None)
        if not isinstance(raw, str):
            return None
        val = raw.strip()
        return val or None

    # ── main event loop ───────────────────────────────────────────────

    async def _main_loop_remote(self) -> None:
        """Loop for remote UI: fixed frame rate; consume snapshots when available.

        Render rate is decoupled from snapshot rate so we get smooth ~20 fps even
        when the backend sends large snapshots slowly (e.g. 1/sec with many bots).
        We wait up to one frame for a snapshot; if none arrives, we re-render the
        last state. This avoids the UI becoming a slideshow when IPC is slow.
        """
        frame_duration = 1.0 / UI_TARGET_FPS
        first_loop = True
        first_snapshot_done = False
        first_render_done = False
        try:
            while self._running:
                if first_loop:
                    first_loop = False
                    if self._checkpoint_cb:
                        self._checkpoint_cb("vision loop started")
                t0 = time.monotonic()
                try:
                    snapshot = await asyncio.wait_for(
                        self._remote_snapshot_queue.get(), timeout=frame_duration
                    )
                    self.update_from_snapshot(snapshot)
                    if not first_snapshot_done and self._checkpoint_cb:
                        first_snapshot_done = True
                        self._checkpoint_cb("first snapshot applied")
                except asyncio.TimeoutError:
                    pass  # no new snapshot this frame; keep last state
                except Exception as e:
                    log.error("Remote UI snapshot error: %s\n%s", e, traceback.format_exc())
                    _write_ui_crash("snapshot", e)
                try:
                    self._frame += 1
                    self.render_frame()
                    if not first_render_done and self._checkpoint_cb:
                        first_render_done = True
                        self._checkpoint_cb("first render done")
                except Exception as e:
                    log.error("Remote UI render error: %s\n%s", e, traceback.format_exc())
                    _write_ui_crash("render", e)
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, frame_duration - elapsed))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Remote UI error: %s\n%s", e, traceback.format_exc())
            _write_ui_crash("main_loop", e)

    async def _main_loop(self) -> None:
        if self._remote_snapshot_queue is not None:
            await self._main_loop_remote()
            return
        if self._bus is None:
            return
        subs = {
            "btc":     await self._bus.subscribe("btc.mid_price"),
            "eth":     await self._bus.subscribe("eth.mid_price"),
            "sol":     await self._bus.subscribe("sol.mid_price"),
            "prob":    await self._bus.subscribe("kalshi.fair_prob"),
            "meta":    await self._bus.subscribe("kalshi.market_metadata"),
            "outcome": await self._bus.subscribe("kalshi.settlement_outcome"),
            "ob":      await self._bus.subscribe("kalshi.orderbook"),
            "fills":   await self._bus.subscribe("kalshi.fills"),
            "orders":  await self._bus.subscribe("kalshi.order_update"),
            "balance": await self._bus.subscribe("kalshi.account_balance"),
            "ws":      await self._bus.subscribe("kalshi.ws.status"),
            "rtt":     await self._bus.subscribe("kalshi.rtt"),
        }

        while self._running:
            try:
                self._frame += 1

                # ── poll queues ───────────────────────────────────────
                dirtied = False
                now = time.time()
                # Shorter budget (20ms) so we yield to event loop and hit ~16 fps with heavy farm load
                budget_deadline = time.perf_counter() + 0.020

                # Priority 1: Control Streams (Process all available)
                # These are critical for bot status, fills, and accounting.
                while not subs["ws"].empty():
                    ws_msg = subs["ws"].get_nowait()
                    status = getattr(ws_msg, "status", "")
                    # Treat "connected" and "reconnecting" as OK so HEALTH doesn't flicker
                    # when the WS briefly disconnects and immediately reconnects.
                    self._ws_connected = status in ("connected", "reconnecting")
                    dirtied = True

                while not subs["rtt"].empty():
                    rtt_msg: KalshiRtt = subs["rtt"].get_nowait()
                    self._kalshi_rtt_ms = rtt_msg.rtt_ms
                    self._kalshi_rtt_source = getattr(rtt_msg, "source", None)
                    dirtied = True

                while not subs["balance"].empty():
                    bal: AccountBalance = subs["balance"].get_nowait()
                    self._balance_usd = bal.balance_usd
                    if self._initial_balance_usd is None:
                        self._initial_balance_usd = bal.balance_usd
                    dirtied = True

                await asyncio.sleep(0)  # yield so farm/WS can run when inline

                # Price queues first so header BTC/ETH/SOL stay fresh when outcomes/fills/orders flood.
                for _ in range(100):
                    if subs["btc"].empty(): break
                    dirtied = True
                    self._prices["BTC"] = subs["btc"].get_nowait().price
                    self._counts["tick"] += 1
                for _ in range(50):
                    if subs["eth"].empty(): break
                    dirtied = True
                    self._prices["ETH"] = subs["eth"].get_nowait().price
                    self._counts["tick"] += 1
                for _ in range(50):
                    if subs["sol"].empty(): break
                    dirtied = True
                    self._prices["SOL"] = subs["sol"].get_nowait().price
                    self._counts["tick"] += 1

                # Process outcomes, fills, and orders with a finite limit per frame so budget reaches OB/prob.
                for _ in range(100):
                    if subs["outcome"].empty(): break
                    out: SettlementOutcome = subs["outcome"].get_nowait()
                    bot_id = self._event_bot_id(out)
                    if bot_id is None:
                        continue

                    # Track per-bot status
                    b_stat = self._bot_stats.setdefault(bot_id, self._empty_bot_stats())
                    b_stat["pnl"] += out.pnl
                    b_stat["gross_pnl"] += float(getattr(out, "gross_pnl", out.pnl))
                    b_stat["fees_usd"] += float(getattr(out, "fees_usd", 0.0))
                    src = getattr(out, "source", "") or ""
                    qty_contracts = float(getattr(out, "quantity_centicx", 0)) / 100.0
                    bucket = _source_bucket(src)
                    if bucket == "scalp":
                        b_stat["pnl_s"] += out.pnl
                        b_stat["qty_s_contracts"] += qty_contracts
                    elif bucket == "arb":
                        b_stat["pnl_a"] += out.pnl
                        b_stat["qty_a_contracts"] += qty_contracts
                    else:
                        b_stat["pnl_e"] += out.pnl
                        b_stat["qty_e_contracts"] += qty_contracts
                    b_stat["trade_count"] += 1
                    if out.pnl >= 0:
                        b_stat["gross_profit"] += out.pnl
                    else:
                        b_stat["gross_loss"] += abs(out.pnl)
                    b_stat["peak_pnl"] = max(b_stat["peak_pnl"], b_stat["pnl"])
                    drawdown = b_stat["peak_pnl"] - b_stat["pnl"]
                    b_stat["max_drawdown"] = max(b_stat["max_drawdown"], drawdown)
                    b_stat["last_active"] = time.time()

                    if self._primary_bot_id and bot_id == self._primary_bot_id:
                        self._session_pnl += out.pnl
                        self._promoted_session_peak_pnl = max(self._promoted_session_peak_pnl, self._session_pnl)
                        self._alltime_pnl += out.pnl
                        if out.won:
                            self._wins += 1
                            self._win_pnl_total += out.pnl
                            self._best_win = max(self._best_win, out.pnl)
                        else:
                            self._losses += 1
                            self._loss_pnl_total += out.pnl
                            self._worst_loss = min(self._worst_loss, out.pnl)

                    if out.won:
                        b_stat["wins"] += 1
                    else:
                        b_stat["losses"] += 1
                    
                    # Only append main/primary bot to history, else it floods
                    if self._primary_bot_id and bot_id == self._primary_bot_id:
                        self._history.appendleft({
                            "won": out.won, "ticker": out.market_ticker,
                            "pnl": out.pnl, "side": out.side,
                            "source": getattr(out, "source", "") or ""
                        })
                    dirtied = True

                for _ in range(500):
                    if time.perf_counter() >= budget_deadline: break
                    if subs["fills"].empty(): break
                    fill: FillEvent = subs["fills"].get_nowait()
                    bot_id = self._event_bot_id(fill)
                    if bot_id is None:
                        continue

                    b_stat = self._bot_stats.setdefault(bot_id, self._empty_bot_stats())
                    b_stat["fills"] += 1
                    bucket = _source_bucket(getattr(fill, "source", "") or "")
                    if bucket == "scalp":
                        b_stat["fills_s"] = b_stat.get("fills_s", 0) + 1
                    elif bucket == "arb":
                        b_stat["fills_a"] = b_stat.get("fills_a", 0) + 1
                    else:
                        b_stat["fills_e"] = b_stat.get("fills_e", 0) + 1
                    b_stat["last_active"] = time.time()
                    
                    if self._primary_bot_id and bot_id == self._primary_bot_id:
                        self._total_fills += 1
                        self._total_contracts += max(1, fill.count // 100)
                    
                    if self._primary_bot_id and bot_id == self._primary_bot_id:
                        self._recent_fills.appendleft({
                            "ticker": fill.market_ticker, "side": fill.side,
                            "price": fill.price_cents, "count": fill.count // 100,
                            "ts": fill.timestamp, "source": getattr(fill, "source", "")
                        })
                        # Feed cascade panel with real ticker stream (5-char abbrev)
                        ticker_abbrev = (fill.market_ticker or "")[:5].upper() or "----"
                        self._cascade_buffer.appendleft(ticker_abbrev)
                    dirtied = True

                for _ in range(200):
                    if time.perf_counter() >= budget_deadline: break
                    if subs["orders"].empty(): break
                    ou: OrderUpdate = subs["orders"].get_nowait()
                    bot_id = self._event_bot_id(ou)
                    if bot_id is None:
                        continue

                    b_stat = self._bot_stats.setdefault(bot_id, self._empty_bot_stats())
                    b_stat["orders"] += 1
                    b_stat["last_active"] = time.time()
                    self._apply_order_bucket_update(ou)
                    
                    if self._primary_bot_id and bot_id == self._primary_bot_id:
                        if ou.status == "placed":
                            self._open_orders += 1
                        elif ou.status in ("filled", "partial_fill", "cancelled", "canceled", "error"):
                            self._open_orders = max(0, self._open_orders - 1)

                for _ in range(200):
                    if time.perf_counter() >= budget_deadline: break
                    if subs["meta"].empty(): break
                    dirtied = True

                await asyncio.sleep(0)  # yield before OB/prob so event loop gets control

                # Priority 2: Data Streams (Optimized batching)
                # We process in smaller batches per stream but cycle through them
                # to ensure Probs don't starve if OB is flooded.
                for _ in range(10): # Cycle through all streams 10 times per frame
                    if time.perf_counter() >= budget_deadline: break
                    
                    # ── OB ──
                    for _ in range(100):
                        if subs["ob"].empty(): break
                        msg = subs["ob"].get_nowait()
                        self._counts["ob"] += 1
                        if st := self._states.get(msg.market_ticker):
                            st.ob_valid = msg.valid
                            if msg.valid:
                                st.ob_had_valid = True
                                st.yes_ask, st.no_ask = msg.implied_yes_ask_cents, msg.implied_no_ask_cents
                                st.yes_bid, st.no_bid = msg.best_yes_bid_cents, msg.best_no_bid_cents
                        dirtied = True
                        
                    # ── Probs ──
                    for _ in range(100):
                        if subs["prob"].empty(): break
                        msg = subs["prob"].get_nowait()
                        self._counts["prob"] += 1
                        if st := self._states.get(msg.market_ticker):
                            st.p_yes = msg.p_yes
                            st.p_yes_hist.append(msg.p_yes)
                        dirtied = True
                    
                    # ── Ticks ──
                    for s in ["btc", "eth", "sol"]:
                        if not subs[s].empty():
                            msg = subs[s].get_nowait()
                            self._prices[s.upper()] = msg.price
                            self._counts["tick"] += 1
                            dirtied = True
                            
                    # ── Meta ──
                    for _ in range(20):
                        if subs["meta"].empty(): break
                        msg = subs["meta"].get_nowait()
                        self._metadata[msg.market_ticker] = msg
                        self._ensure_state(msg)
                        self._counts["mkt"] += 1
                        dirtied = True

                # Sample prices for sparklines
                if self._frame % 5 == 0:
                    for a in ["BTC", "ETH", "SOL"]:
                        if self._prices[a] > 0: self._price_hist[a].append(self._prices[a])

                # ── render ────────────────────────────────────────────
                min_render_interval = 1.0 / UI_MAX_RENDER_HZ
                # Force render every 2s for heartbeat/uptime even if no data changed
                force_heartbeat = (now - self._last_render_ts) >= 2.0
                
                if (dirtied or force_heartbeat) and (now - self._last_render_ts) >= min_render_interval:
                    self._render()
                    self._last_render_ts = now

                await asyncio.sleep(1.0 / UI_TARGET_FPS)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                log.error(f"UI error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1.0)

    # ── market selection ──────────────────────────────────────────────

    def _distance_to_spot(self, st: TickerState) -> float:
        """
        Normalized distance of this market to current spot (near-money measure).
        Binary: |spot - strike| / spot. Range: |spot - mid| / spot, or 0 if spot in [floor, cap].
        Matches runner _rank_near_money for consistency. Returns 999 if no spot or invalid range.
        """
        spot = self._prices.get(st.asset, 0.0) or 0.0
        if spot <= 0:
            return 999.0
        if st.is_range:
            lo, hi = st.strike_floor, st.strike_cap
            if lo is None or hi is None:
                return 999.0
            if lo <= spot <= hi:
                return 0.0
            mid = (lo + hi) / 2
            return abs(spot - mid) / spot
        if st.strike <= 0:
            return 999.0
        return abs(spot - st.strike) / spot

    def _display_assets(self) -> List[str]:
        assets = []
        seen_assets = sorted({s.asset for s in self._states.values() if getattr(s, "asset", None)})
        for asset in ("BTC", "ETH", "SOL"):
            if asset in seen_assets or self._prices.get(asset, 0.0) > 0:
                assets.append(asset)
        if assets:
            return [a for a in assets if a != "SOL"] if "SOL" not in seen_assets else assets
        return list(DEFAULT_DISPLAY_ASSETS)

    def _signal_label_for_state(self, st: TickerState, arrow: str) -> str:
        if not st.ob_valid:
            if st.ob_had_valid:
                return f"{GRAY}STALE{RESET}"
            return f"{DKGRAY}WAIT{RESET}"

        window_label = st.window_label
        ask_cents = st.best_ask
        edge = st.best_edge
        hold_ok = (
            _ui_hold_family_allowed(st.asset, window_label)
            and UI_HOLD_MIN_ENTRY_CENTS <= ask_cents <= UI_HOLD_MAX_ENTRY_CENTS
            and not (st.best_side == "NO" and ask_cents >= UI_NO_AVOID_ABOVE_CENTS)
            and edge >= UI_HOLD_MIN_EDGE_CENTS
        )
        scalp_ok = (
            _ui_scalp_family_allowed(st.asset, window_label)
            and UI_SCALP_MIN_ENTRY_CENTS <= ask_cents <= UI_SCALP_MAX_ENTRY_CENTS
            and st.spread <= UI_SCALP_MAX_SPREAD_CENTS
            and edge >= UI_SCALP_MIN_EDGE_CENTS
        )
        if hold_ok or scalp_ok:
            s_col = GREEN if st.best_side == "YES" else RED
            conviction = "!!!" if edge >= 20 else "!!" if edge >= 12 else "!"
            return f"{s_col}{BOLD}{arrow} BUY {st.best_side} {conviction}{RESET}"
        return f"{DKGRAY}PASS{RESET}"

    def _best_per_type(self) -> Dict[str, Dict[str, Optional[TickerState]]]:
        """
        Return best market per (asset, window_type).
        Selection: within each (asset, window) prefer live OB first, then near-money.
        This keeps displayed rows aligned with actively updating orderbooks.
        """
        now = time.time()
        if now - self._last_best_markets_update < 0.5 and self._best_markets_cache:
            # Invalidate if any displayed market has expired so we don't show EMPTY until next 0.5s tick
            invalidated = False
            for asset_bucket in self._best_markets_cache.values():
                for st in asset_bucket.values():
                    if st is not None and st.time_remaining_raw <= 0:
                        self._best_markets_cache = {}
                        invalidated = True
                        break
                if invalidated:
                    break
            if not invalidated and self._best_markets_cache:
                return self._best_markets_cache

        result: Dict[str, Dict[str, Optional[TickerState]]] = {
            asset: {"15min": None, "60min": None, "Range": None}
            for asset in self._display_assets()
        }

        def _score(s: TickerState) -> tuple:
            dist = self._distance_to_spot(s)
            # Prefer: live OB, then stale-known OB, then near money, then soonest expiry, then edge.
            ob_bucket = 0 if s.ob_valid else (1 if s.ob_had_valid else 2)
            return (
                ob_bucket,
                dist,
                s.exp_ts if s.exp_ts > 0 else 1e15,
                -s.best_edge,
            )

        states_iter = self._states.values()
        for st in states_iter:
            if st.time_remaining_raw <= 0:
                continue
            wl = st.window_label
            # Only consider markets expiring within the hour for 60m/range display.
            # This avoids selecting next-hour/far-future contracts that show as "1h00m"
            # and often lack active orderbook updates for the current trading window.
            RANGE_MAX_MINUTES = 60
            if wl in {"60min", "Range"}:
                ttr_s = st.time_remaining_raw
                if ttr_s > RANGE_MAX_MINUTES * 60:
                    continue
            bucket = result.get(st.asset, {})
            if wl not in bucket:
                continue
            cur = bucket[wl]
            if cur is None or _score(st) < _score(cur):
                bucket[wl] = st

        self._best_markets_cache = result
        self._last_best_markets_update = now
        return result

    def _count_active_markets(self) -> int:
        """Count tradeable markets: has valid expiry and not yet expired."""
        now = time.time()
        return sum(1 for s in self._states.values() if s.exp_ts > 0 and s.exp_ts > now)

    def _refresh_promoted_metadata(self) -> None:
        now = time.time()
        if now - self._promoted_meta_load_ts < PROMOTED_METADATA_CACHE_TTL:
            return
        self._promoted_meta_load_ts = now
        self._promoted_meta = {}
        self._lifetime_max_fills = 0
        try:
            if PROMOTED_JSON_PATH.exists():
                payload = json.loads(PROMOTED_JSON_PATH.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._promoted_meta = payload
        except Exception:
            self._promoted_meta = {}
        try:
            if LIFETIME_JSON_PATH.exists():
                payload = json.loads(LIFETIME_JSON_PATH.read_text(encoding="utf-8"))
                bots = payload.get("bots", {}) if isinstance(payload, dict) else {}
                if isinstance(bots, dict) and bots:
                    self._lifetime_max_fills = max(
                        int((stats or {}).get("fills", 0) or 0)
                        for stats in bots.values()
                        if isinstance(stats, dict)
                    )
        except Exception:
            self._lifetime_max_fills = 0

    def _promoted_params_text(self, params: Dict[str, Any], width: int) -> str:
        if not params:
            return "Params unavailable"
        parts = [f"{key}={value}" for key, value in sorted(params.items())]
        text = "  ".join(parts)
        return text if len(text) <= width else text[: max(0, width - 3)] + "..."

    def _promoted_session_drawdown_pct(self) -> float:
        if not self._primary_bot_id:
            return 0.0
        stats = self._bot_stats.get(self._primary_bot_id, {})
        bankroll = float(stats.get("start_equity", 5000.0) or 5000.0)
        if bankroll <= 0:
            bankroll = 5000.0
        current_drawdown = max(0.0, self._promoted_session_peak_pnl - self._session_pnl)
        return current_drawdown / bankroll if bankroll > 0 else 0.0

    def _append_promoted_bot_section(self, lines: List[str], width: int) -> None:
        self._refresh_promoted_metadata()
        accent = _header_lights(self._frame)
        lines.append(
            f" {AMBER}{BOLD}══ PROMOTED BOT{RESET}"
            f"  {WHITE}{accent}{RESET}"
            f"{DKGRAY}{'═' * max(0, width - 20)}{RESET}{CLR}"
        )
        promoted_id = self._primary_bot_id or str(self._promoted_meta.get("bot_id") or "").strip() or None
        if not promoted_id:
            lines.append(f"  {GRAY}No bot promoted yet — farm accumulating data{RESET}{CLR}")
            return

        promoted_at = str(self._promoted_meta.get("promotion_timestamp") or "n/a")
        lifetime_stats = dict(self._promoted_meta.get("lifetime_stats") or {})
        params = dict(self._promoted_meta.get("params") or lifetime_stats.get("params") or {})
        score = self._promoted_meta.get("robustness_score")
        score_str = f"{float(score):.2f}" if isinstance(score, (int, float)) else "n/a"
        fills = int(lifetime_stats.get("fills", 0) or 0)
        pnl = float(lifetime_stats.get("total_pnl", 0.0) or 0.0)
        win_rate = float(lifetime_stats.get("win_rate", 0.0) or 0.0)
        pnl_color = GREEN if pnl >= 0 else RED
        drawdown_pct = self._promoted_session_drawdown_pct()
        drawdown_warn = ""
        if drawdown_pct > 0.03:
            drawdown_warn = f"  {YELLOW}{BOLD}WARN drawdown {drawdown_pct:.1%} of bankroll{RESET}"

        lines.append(
            f"  {WHITE}bot_id {AMBER}{BOLD}{promoted_id}{RESET}"
            f"  {GRAY}│{RESET}  {WHITE}Promoted {CYAN}{promoted_at}{RESET}"
            f"  {GRAY}│{RESET}  {WHITE}Robust {AMBER}{BOLD}{score_str}{RESET}"
            f"{drawdown_warn}{CLR}"
        )
        lines.append(
            f"  {WHITE}Lifetime fills {AMBER}{fills}{RESET}"
            f"  {GRAY}│{RESET}  {WHITE}Lifetime PnL {pnl_color}{BOLD}${pnl:+.2f}{RESET}"
            f"  {GRAY}│{RESET}  {WHITE}Lifetime WR {AMBER}{win_rate:.1%}{RESET}{CLR}"
        )
        lines.append(f"  {DKGRAY}{self._promoted_params_text(params, max(20, width - 4))}{RESET}{CLR}")

    def _append_leaderboard(self, lines: List[str], width: int, rotator: str) -> None:
        lines.append(f" {GRAY}{'─' * width}{RESET}{CLR}")
        unique_str = f"      {AMBER}Unique {self._unique_config_count}{RESET}" if self._unique_config_count is not None else ""
        header_lights = _header_lights(self._frame)
        lines.append(
            f" {ORANGE}{BOLD}══ DWARF RANKING (TOP 5 + BOTTOM 5){RESET}"
            f"  {WHITE}{header_lights}{RESET}"
            f"{unique_str}"
            f"{CLR}"
        )
        lines.append(f" {GRAY}{'─' * width}{RESET}{CLR}")
        # 3-column grid + stream labels: 5 rows stacked vertically, KXBTC/KXETH/KXSOL under prongs.
        grid = self._render_grid()
        stream_keys = list(self._states.keys())[:3]
        if not stream_keys:
            stream_keys = ["KXBTC", "KXETH", "KXSOL"]
        stream_cells = " ".join(f"│{(t or '')[:5].upper():<5}│" for t in stream_keys)
        grid_row_len = len(grid[0]) if grid else 17
        pad = max(0, (width - grid_row_len) // 2)
        strm_offset = (len(grid[0]) - 4) // 2 if grid else 6
        lines.append(f" {" " * (pad + strm_offset)}{CYAN}STRM{RESET}{CLR}")
        for row in grid:
            lines.append(f" {" " * pad}{CYAN}{row}{RESET}{CLR}")
        stream_pad = max(0, (width - len(stream_cells)) // 2)
        lines.append(f" {" " * stream_pad}{WHITE}{stream_cells}{RESET}{CLR}")
        lines.append(
            f"  {BOLD}{DKGRAY}"
            f"{'Bot Name':<22}"
            f"{'Robust':>7}  "
            f"{'Net PnL':>9}  "
            f"{'E/S/A PnL':>23}  "
            f"{'WR':>6}  "
            f"{'E/S/A Fills':>14}  "
            f"{'Bought':>8}  "
            f"{'Trd':>4} "
            f"{'Gen':>3}  "
            f"{RESET}{CLR}"
        )

        now = time.time()
        if now - self._last_leaderboard_update >= LEADERBOARD_CACHE_TTL or not self._bot_leaderboard_cache:
            active_bots = []
            bot_items = self._bot_stats.items()
            for bid, stats in bot_items:
                if 5000.0 + stats.get("pnl", 0.0) <= 0.0:
                    continue  # exclude busted bots
                active_bots.append((bid, stats))

            enriched: List[tuple[str, Dict[str, Any], float]] = []
            for bid, stats in active_bots:
                score = calculate_robustness_score(stats)
                enriched.append((bid, stats, score))

            enriched.sort(key=lambda item: (item[2], item[1].get("pnl", 0.0)), reverse=True)
            self._bot_leaderboard_cache = enriched
            self._last_leaderboard_update = now

        top_20 = self._bot_leaderboard_cache[:5]
        bottom_5 = self._bot_leaderboard_cache[-5:] if len(self._bot_leaderboard_cache) > 5 else []
        display_rows = top_20 + bottom_5

        trail_len = 16
        for idx, (bid, stats, robust) in enumerate(display_rows):
            if idx == 0:
                lines.append(f"  {CYAN}{BOLD}TOP 5{RESET}{CLR}")
            elif idx == len(top_20) and bottom_5:
                lines.append(f"  {GRAY}{'-' * max(0, width - 2)}{RESET}{CLR}")
                lines.append(
                    f"  {DKGRAY}diag{RESET} {CYAN}{rotator}{RESET}  "
                    f"{WHITE}{_wave_segment(self._frame, min(28, max(8, width // 4)), phase=6)}{RESET}{CLR}"
                )
                lines.append(f"  {CYAN}{BOLD}BOTTOM 5{RESET}{CLR}")
            pnl = stats["pnl"]
            pnl_e = stats.get("pnl_e", 0.0)
            pnl_s = stats.get("pnl_s", 0.0)
            pnl_a = stats.get("pnl_a", 0.0)
            tot = stats.get("trade_count", stats["wins"] + stats["losses"])
            wr = (stats["wins"] / tot * 100) if tot > 0 else 0.0
            gross_profit = stats.get("gross_profit", 0.0)
            gross_loss = stats.get("gross_loss", 0.0)
            pnl_c = GREEN if pnl >= 0 else RED
            pnl_e_c = GREEN if pnl_e >= 0 else RED
            pnl_s_c = GREEN if pnl_s >= 0 else RED
            pnl_a_c = GREEN if pnl_a >= 0 else RED
            alpha_c = GREEN if robust >= 40 else (AMBER if robust >= 20 else RED)
            signif_flag = "*" if tot < MIN_SETTLED_TRADES_FOR_ALPHA else " "
            fills_e = stats.get("fills_e", 0)
            fills_s = stats.get("fills_s", 0)
            fills_a = stats.get("fills_a", 0)
            buy_contracts = float(stats.get("buy_contracts", 0.0))
            gen = int(stats.get("generation", 0))
            is_promoted = bool(self._primary_bot_id and bid == self._primary_bot_id)
            marker = "[★] " if is_promoted else "    "
            name_color = AMBER if is_promoted else WHITE

            # Animated "worker bot" trail — one bright dot per row; slightly different speed per row.
            step = 4 + (idx % 3)
            pos = (self._frame * step // 4 + idx * 2) % trail_len
            trail = "".join("·" if j != pos else "●" for j in range(trail_len))

            lines.append(
                f"  {name_color}{marker}{bid:<18}{RESET}"
                f"{alpha_c}{BOLD}{robust:>7.2f}{RESET}{signif_flag} "
                f"{pnl_c}{BOLD}{pnl:>+9.2f}{RESET}  "
                f"{pnl_e_c}{pnl_e:>+7.1f}{RESET}/{pnl_s_c}{pnl_s:>+7.1f}{RESET}/{pnl_a_c}{pnl_a:>+7.1f}{RESET}  "
                f"{AMBER}{wr:>5.1f}%{RESET}  "
                f"{WHITE}{fills_e:>4}/{fills_s:>4}/{fills_a:<4}{RESET}"
                f"  "
                f"{WHITE}{buy_contracts:>8.1f}{RESET}"
                f"  "
                f"{WHITE}{tot:>4}{RESET} "
                f"{DKGRAY}{gen:>3}{RESET}  "
                f"{CYAN}[{trail}]{RESET}"
                f"{CLR}"
            )

        if not top_20:
            lines.append(f"  {GRAY}Awaiting initial bot activity to populate leaderboard...{RESET}{CLR}")
        else:
            lines.append(f"  {DIM}{GRAY}* Robust score ranks worst-case execution resilience; low samples are heavily penalized{RESET}{CLR}")

    # ── NGE panel renderers (all driven by real data) ───────────────────

    def _render_scan_sweep(self) -> List[str]:
        """Left-side diagnostic: sweep position from WS + market count (real state)."""
        n_markets = len(self._states)
        ws_off = 1 if self._ws_connected else 0
        idx = (self._panel_scan + n_markets + ws_off) % len(_SCAN_SWEEP_FRAMES)
        return list(_SCAN_SWEEP_FRAMES[idx])

    def _render_radar(self) -> str:
        """Target acquisition: radar position from frame; targets = actionable market count."""
        idx = self._panel_radar % len(_RADAR_FRAMES)
        n_actionable = sum(1 for s in self._states.values() if s.is_actionable())
        # Optional: color or width from n_actionable (real targets locked)
        return _RADAR_FRAMES[idx]

    def _render_telemetry_bars(self) -> str:
        """Vertical bar telemetry: 8 levels from real rates (OB/s, ticks/s, fills/hr, etc.)."""
        up_s = max(1, int(time.time() - self._start_time))
        ob_s = self._counts["ob"] / up_s
        tick_s = self._counts["tick"] / up_s
        prob_s = self._counts["prob"] / up_s
        fills_hr = (self._total_fills / (up_s / 3600.0)) if up_s else 0
        # Normalize to 0–8 for bar height; scale so typical values land in mid range
        bars = [
            min(8, int(ob_s / 4)),
            min(8, int(tick_s / 10)),
            min(8, int(prob_s / 4)),
            min(8, int(fills_hr / 5)),
            min(8, self._open_orders),
            min(8, self._win_streak),
            min(8, self._loss_streak),
            min(8, len(self._states) // 3),
        ]
        return "│" + "".join(_BAR_LEVELS[h] for h in bars) + "│"

    def _render_scope(self) -> List[str]:
        """Oscilloscope: wave from real BTC price history or first market p_yes."""
        data = list(self._price_hist.get("BTC", deque(maxlen=12)))
        if len(data) < 3:
            data = list(self._price_hist.get("ETH", deque(maxlen=12)))
        if len(data) < 3:
            for st in self._states.values():
                data = list(st.p_yes_hist)
                if len(data) >= 3:
                    break
        if len(data) < 3:
            data = [0.5, 0.5, 0.5]
        lo, hi = min(data), max(data)
        rng = (hi - lo) or 1
        indices = [min(len(_SCOPE_WAVE) - 1, int((v - lo) / rng * (len(_SCOPE_WAVE) - 1))) for v in data[-10:]]
        wave_str = "".join(_SCOPE_WAVE[i] for i in indices).ljust(10, " ")[:10]
        # Three scan lines with phase offset (Evangelion-style)
        l1 = wave_str
        l2 = (wave_str[1:] + wave_str[0]) if len(wave_str) >= 10 else wave_str
        l3 = (wave_str[2:] + wave_str[:2]) if len(wave_str) >= 10 else wave_str
        return [
            "┌────────────┐",
            "│" + l1 + "│",
            "│" + l2 + "│",
            "│" + l3 + "│",
            "└────────────┘",
        ]

    def _render_grid(self) -> List[str]:
        """Subsystem diagnostics: WS vs data; flicker reflects real state."""
        idx = (self._panel_grid + (1 if self._ws_connected else 0)) % len(_GRID_FRAMES)
        return list(_GRID_FRAMES[idx])

    def _render_frame_indicator(self) -> str:
        """Status module: ACTIVE. animation; label reflects SIM/LIVE (real mode)."""
        idx = self._panel_frame % len(_FRAME_ACTIVE)
        return _FRAME_ACTIVE[idx].strip()

    def _render_cascade(self) -> List[str]:
        """Streaming system data: real ticker snippets from fills; pad from active markets."""
        rows = []
        cascade_src = list(self._cascade_buffer)
        if len(cascade_src) < 5:
            for t in list(self._states.keys())[: 5 - len(cascade_src)]:
                cascade_src.append((t or "")[:5].upper() or "-----")
        for i in range(5):
            j = (i + self._panel_cascade) % max(1, len(cascade_src))
            raw = cascade_src[j] if j < len(cascade_src) else "     "
            snippet = (raw if isinstance(raw, str) else str(raw))[:5].ljust(5)
            rows.append("│" + snippet + "│")
        return rows

    def _render_background_line(self) -> List[str]:
        """Low-frequency global fill sweep (frame-based, no fake data)."""
        idx = (self._panel_bg // 15) % len(_BACKGROUND_FRAMES)
        return list(_BACKGROUND_FRAMES[idx])

    def _render_population_epoch_indicator(self, width: int, now: float, frame: int) -> List[str]:
        """Render an epoch/reseed countdown block for population manager."""
        if not self._population_enabled:
            return [f"  {DKGRAY}EPOCH WINDOW  disabled{RESET}"]

        epoch_sec = max(1.0, float(self._population_epoch_minutes) * 60.0)
        anchor = self._population_last_epoch_ts if self._population_last_epoch_ts > 0 else self._runtime_start_time
        elapsed = max(0.0, now - anchor)
        remaining = max(0.0, epoch_sec - (elapsed % epoch_sec))
        glyphs = ["◈", "◇", "◆", "◉", "◎"]
        glyph = glyphs[frame % len(glyphs)]

        # 10-wide bar shrinks toward reseed.
        filled = max(0, min(10, int(round((remaining / epoch_sec) * 10))))
        bar = ("█" * filled) + ("░" * (10 - filled))
        mm = int(remaining // 60)
        ss = int(remaining % 60)
        ttxt = f"{mm:02d}:{ss:02d}"

        lines = [
            f"  {CYAN}{BOLD}EPOCH WINDOW{RESET}  {WHITE}{glyph}{RESET}",
            f"  {WHITE}T-{ttxt}{RESET}  {CYAN}{bar}{RESET}  {AMBER}Gen {self._population_generation}{RESET}",
        ]

        # Final 5-minute dramatic countdown ladder in the style requested.
        if remaining <= 5 * 60:
            mins_left = max(0, int((remaining + 59) // 60))
            ladder = [
                ("T-05", "██████████", "◈"),
                ("T-04", "█████████░", "◇"),
                ("T-03", "████████░░", "◆"),
                ("T-02", "███████░░░", "◉"),
                ("T-01", "██████░░░░", "◎"),
            ]
            for i, (lbl, ladder_bar, lg) in enumerate(ladder, start=5):
                active = mins_left <= i
                c = CYAN if active else DKGRAY
                lines.append(f"  {WHITE}{lbl}{RESET}  {c}{ladder_bar}{RESET}  {WHITE}{lg}{RESET}")
            if remaining <= 1.0:
                lines.append(f"  {AMBER}{BOLD}T-00  RESEED{RESET}")

        reseed_note = (
            f"R:{self._population_reseed_total} "
            f"(E:{self._population_reseed_epoch} D:{self._population_reseed_drawdown})"
        )
        lines.append(f"  {DKGRAY}{reseed_note}{RESET}")
        return lines

    # ── rendering ──────────────────────────────────────────────

    def _render(self) -> None:
        frame  = self._frame
        spin   = _SPINNERS[frame % 4]
        arrow  = _ARROWS[frame % 2]
        now = time.time()
        up_s   = int(now - self._start_time)
        mm, ss = divmod(up_s, 60)
        hh, mm = divmod(mm, 60)
        up_str = f"{hh}h{mm:02d}m" if hh else f"{mm}m{ss:02d}s"
        runtime_s = max(1, int(now - self._runtime_start_time))
        fps    = frame / runtime_s

        # Advance NGE panels at different rates (async feel)
        if frame % 4 == 0:
            self._panel_scan += 1
        if frame % 3 == 0:
            self._panel_radar += 1
        self._panel_bars = frame
        if frame % 2 == 0:
            self._panel_scope += 1
        if frame % 5 == 0:
            self._panel_grid += 1
        self._panel_frame = frame
        if frame % 15 == 0:
            self._panel_bg += 1
        if frame % 2 == 0:
            self._panel_cascade += 1

        mode_str = (
            f"{RED}{BOLD}◉ LIVE{RESET}"
            if not self._dry_run
            else f"{CYAN}{BOLD}◈ SIM{RESET}"
        )
        bal_str = (
            f"{GRAY}┃{RESET}  {WHITE}Bal {RESET}{BOLD}{GREEN}${self._balance_usd:,.2f}{RESET}"
            if self._balance_usd is not None
            else ""
        )

        # Animated wave for header decoration
        wave_off = frame % len(_WAVE)
        wave = "".join(_WAVE[(i + wave_off) % len(_WAVE)] for i in range(14))

        L: List[str] = []

        # Get current terminal size (cached to avoid syscalls every frame).
        # In remote (separate-process) mode, avoid probing the Windows subprocess
        # console directly. Use a larger fixed canvas so the leaderboard is not
        # truncated to a fake 40-line window.
        if self._remote_snapshot_queue is not None:
            cols, rows = DEFAULT_W, DEFAULT_REMOTE_ROWS
        else:
            now_ts = time.time()
            if now_ts - self._terminal_size_ts >= TERMINAL_SIZE_CACHE_TTL:
                try:
                    sz = shutil.get_terminal_size((DEFAULT_W, 40))
                    cols, rows = max(80, min(200, sz.columns)), max(24, min(80, sz.lines))
                    self._terminal_size = (cols, rows)
                except (OSError, AttributeError):
                    self._terminal_size = (DEFAULT_W, 40)
                self._terminal_size_ts = now_ts
            cols, rows = self._terminal_size
        # Use a wider layout so top sections match leaderboard span.
        W = min(max(UI_MIN_LAYOUT_WIDTH, cols - 2), 140)

        # Adjust list counts based on rows.
        # Markets ~14, Stats 5, DIAG block ~9, header/prices/telemetry/orders/history ~8.
        available = max(1, rows - 36)
        max_fills = max(1, available * 2 // 3)   # 2/3 of remaining for fills
        max_hist  = max(1, available // 3)        # 1/3 of remaining for history

        # ╔══ HEADER ══════════════════════════════════════════════════════╗
        health_color = GREEN if self._ws_connected else RED
        health_text  = "OK" if self._ws_connected else "ERR"
        health_str   = f"{health_color}[HEALTH:{health_text}]{RESET}"

        rotator = _ROTATORS[frame % len(_ROTATORS)]
        lights = _LIGHTS[frame % len(_LIGHTS)]

        L.append(
            f"{ORANGE}{BOLD} {spin} ARGUS·VISION{RESET}"
            f"  {mode_str}"
            f"  {GRAY}┃{RESET}  Up {BOLD}{WHITE}{up_str}{RESET}"
            f"  {DKGRAY}Build {self._ui_build_stamp}{RESET}"
            f"  {bal_str}"
            f"  {health_str}  {CYAN}{rotator} {lights}{RESET}"
            f"  {WHITE}{wave}{RESET}"
            f"{CLR}"
        )

        # Prices + sparklines
        price_parts = []
        for a in self._display_assets():
            p     = self._prices[a]
            spark = _sparkline(self._price_hist[a], width=8)
            if p > 0:
                price_parts.append(
                    f"{BOLD}{WHITE}{a}{RESET} {ORANGE}${p:,.0f}{RESET}"
                    f" {WHITE}{spark}{RESET}"
                )
            else:
                price_parts.append(f"{GRAY}{a} ···{RESET}")
        sep = f"  {GRAY}│{RESET}  "
        L.append(f" {sep.join(price_parts)}{CLR}")

        # Telemetry (Mkts, Probs, OB, Kalshi RTT, fps). Prefer WS when available (order latency).
        if self._kalshi_rtt_ms is not None:
            src = (self._kalshi_rtt_source or "rest").upper()
            rtt_str = f"Kalshi {self._kalshi_rtt_ms:.0f}ms ({src})"
        else:
            rtt_str = "Kalshi —"
        L.append(
            f" {WHITE}Mkts {CYAN}{self._counts['mkt']:,}{RESET}"
            f"  {WHITE}Probs {CYAN}{self._counts['prob']:,}{RESET}"
            f"  {WHITE}OB {CYAN}{self._counts['ob']:,}{RESET}"
            f"  {AMBER}{rtt_str}{RESET}"
            f"  {WHITE}fps {CYAN}{fps:.0f}{RESET}{CLR}"
        )

        self._append_promoted_bot_section(L, W)

        # ── MARKETS ═══════════════════════════════════════════════════
        bot_lbl = self._primary_bot_id if self._primary_bot_id else "—"
        header_accent = _header_lights(frame)
        L.append(
            f" {ORANGE}{BOLD}══ MARKETS{RESET}"
            f"  {WHITE}{header_accent}{RESET}"
            f"{DKGRAY}{'═' * max(0, W - 13)}{RESET}{CLR}"
        )
        L.append(f" {DKGRAY}Bot: {AMBER}{BOLD}{bot_lbl}{RESET}{CLR}")
        L.append(
            f" {BOLD}{DKGRAY}"
            f"{'Asset':<6}{'Window':<7}{'Strike (Ref)':<24} "
            f"{'Fair':>4}  {'History':^10} {'Ask':>5} {'Edge':>5} {'Ord':>5} "
            f"{'Expires':>8} Signal"
            f"{RESET}{CLR}"
        )
        L.append(f" {DKGRAY}{'─' * W}{RESET}{CLR}")

        best = self._best_per_type()
        display_assets = self._display_assets()
        for asset in display_assets:
            for wt in ["15min", "60min"]:
                st = best[asset].get(wt)

                if st is None:
                    # Maintain alignment even for missing markets
                    L.append(
                        f" {WHITE}{asset:<6}{RESET}"
                        f"{GRAY}{wt:<7}{'—' * 24}{RESET} "
                        f"{GRAY}{'---':>4}   {'[      ]':^10} {'---':>5} {'---':>5} {'0':>5} "
                        f"{'--:--':>8} {DKGRAY}EMPTY{RESET}"
                        f"{CLR}"
                    )
                    continue

                fair_str = f"{st.p_yes:.0%}"
                prob_spark = _sparkline(st.p_yes_hist, width=8)
                
                # Color based on probability level
                p_col = (
                    CYAN if 0.40 <= st.p_yes <= 0.60
                    else GREEN if st.p_yes > 0.60
                    else RED
                )

                if st.ob_valid:
                    ask_str = f"{st.best_ask:02d}¢"
                    a_col   = WHITE
                elif st.ob_had_valid:
                    ask_str = f"~{st.best_ask:02d}¢"   # stale but known
                    a_col   = GRAY
                else:
                    ask_str = "---"
                    a_col   = GRAY

                edge     = st.best_edge
                if st.ob_valid:
                    edge_str = f"{edge:+d}¢"
                    e_col = (
                        GREEN if edge >= EDGE_THRESHOLD_CENTS
                        else AMBER if edge > 0
                        else GRAY
                    )
                elif st.ob_had_valid:
                    edge_str = f"~{edge:+d}¢"  # stale
                    e_col    = GRAY
                else:
                    edge_str = "---"
                    e_col    = GRAY

                # Reference price comparison: Spot vs Strike
                # We pad the PLAIN TEXT first because ANSI codes break f-string alignment
                strike_txt = st.strike_display
                spot = self._prices[asset]
                plausible_strike = (
                    spot <= 0
                    or st.is_range
                    or st.is_directional
                    or st.strike <= 0
                    or abs(st.strike - spot) <= max(250.0, spot * 0.5)
                )
                if not plausible_strike:
                    strike_txt = "---"
                if spot > 0 and plausible_strike and st.strike > 0:
                    delta_raw = spot - st.strike
                    d_col = GREEN if delta_raw > 0 else RED
                    delta_txt = f"({delta_raw:+.0f})"
                    # Visually: strike_txt (16) + delta_txt (8) = 24 chars
                    s_part = f"{CYAN}{strike_txt:<16}{RESET}"
                    d_part = f"{d_col}{delta_txt:<8}{RESET}"
                else:
                    s_part = f"{CYAN}{strike_txt:<16}{RESET}"
                    d_part = f"{' ':<8}"

                strike_col = f"{s_part}{d_part}"

                # Signal display (only when a bot is promoted; signals align with bot strategy)
                if self._primary_bot_id:
                    sig = self._signal_label_for_state(st, arrow)
                else:
                    sig = f"{DKGRAY}—{RESET}"

                bucket_key = st.ticker
                order_count = self._market_fill_counts.get(bucket_key, 0)

                L.append(
                    f" {BOLD}{WHITE}{asset:<6}{RESET}"
                    f"{WHITE}{wt:<7}{RESET}"
                    f"{strike_col} "
                    f"{p_col}{fair_str:>4}{RESET}  "
                    f"{p_col}[{prob_spark}]{RESET} "
                    f"{a_col}{ask_str:>5}{RESET} "
                    f"{e_col}{edge_str:>5}{RESET} "
                    f"{WHITE}{order_count:>5}{RESET} "
                    f"{AMBER}{st.time_remaining:>8}{RESET} "
                    f"{sig}"
                    f"{CLR}"
                )

            # Animated separator: circle flows at different speeds per asset (slower than other UI)
            _asset_idx = display_assets.index(asset)
            sep_phase = _asset_idx * 11
            sep_speed = (4, 5, 6)[min(_asset_idx, 2)]
            L.append(f" {DKGRAY}{_markets_sep_line(frame, W, sep_phase, sep_speed)}{RESET}{CLR}")

        # ── STATS ═════════════════════════════════════════════════════
        at_c = GREEN if self._alltime_pnl >= 0 else RED  # used by ORDERS header when no primary
        stats_lights = _header_lights(frame)
        L.append(
            f" {CYAN}{BOLD}══ STATS{RESET}"
            f"  {WHITE}{stats_lights}{RESET}"
            f"{DKGRAY}{'═' * max(0, W - 13)}{RESET}{CLR}"
        )
        for pl in self._render_population_epoch_indicator(W, now, frame):
            L.append(f"{pl}{CLR}")
        _no_promoted = not self._primary_bot_id or (self._primary_bot_id or "").strip() == "default"
        if _no_promoted:
            L.append(f"  {GRAY}No promoted bot session yet — farm is still accumulating promotion evidence{RESET}{CLR}")
        else:
            probe = self._render_scan_sweep()
            up_h         = max(0.001, up_s / 3600)
            per_hour     = self._alltime_pnl / up_h
            per_day      = per_hour * 24
            fills_per_hr = self._total_fills / up_h
            at_c   = GREEN if self._alltime_pnl  >= 0 else RED
            sess_c = GREEN if self._session_pnl  >= 0 else RED
            ph_c   = GREEN if per_hour >= 0 else RED
            pd_c   = GREEN if per_day  >= 0 else RED
            # Row 1 — Balance
            if self._balance_usd is not None:
                net      = self._balance_usd - (self._initial_balance_usd or self._balance_usd)
                net_c    = GREEN if net >= 0 else RED
                init_str = f"${self._initial_balance_usd:,.2f}" if self._initial_balance_usd else "---"
                L.append(
                    f" {AMBER}{probe[0]}{RESET}  {WHITE}Balance  {GREEN}{BOLD}${self._balance_usd:,.2f}{RESET}"
                    f"  {GRAY}│{RESET}  {WHITE}Principal  {AMBER}{BOLD}{init_str}{RESET}"
                    f"  {GRAY}│{RESET}  {WHITE}Net  {net_c}{BOLD}${net:+.2f}{RESET}"
                    f"{CLR}"
                )
            else:
                L.append(f" {AMBER}{probe[0]}{RESET}  {GRAY}Awaiting balance update…{RESET}{CLR}")
            # Row 2 — Rates
            L.append(
                f" {AMBER}{probe[1]}{RESET}  {WHITE}$/Hour  {ph_c}{BOLD}{per_hour:+.2f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}$/Day  {pd_c}{BOLD}{per_day:+.2f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Session  {sess_c}{BOLD}${self._session_pnl:+.2f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Fills/Hr  {AMBER}{fills_per_hr:.1f}{RESET}"
                f"{CLR}"
            )
            # Row 3 — Trade quality
            avg_win  = (self._win_pnl_total  / self._wins)   if self._wins   > 0 else 0.0
            avg_loss = (self._loss_pnl_total / self._losses) if self._losses > 0 else 0.0
            best_str  = f"${self._best_win:+.2f}"   if self._wins   > 0 else "---"
            worst_str = f"${self._worst_loss:+.2f}" if self._losses > 0 else "---"
            L.append(
                f" {AMBER}{probe[2]}{RESET}  {WHITE}Avg Win  {GREEN}{avg_win:+.2f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Avg Loss  {RED}{avg_loss:+.2f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Best  {GREEN}{best_str}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Worst  {RED}{worst_str}{RESET}"
                f"{CLR}"
            )
            # Row 4 — Activity
            n_active = self._count_active_markets()
            ob_per_s  = self._counts["ob"]   / max(1, up_s)
            tik_per_s = self._counts["tick"] / max(1, up_s)
            L.append(
                f" {AMBER}{probe[3]}{RESET}  {WHITE}Contracts  {AMBER}{self._total_contracts}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Active Mkts  {AMBER}{n_active}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}OB/s  {DKGRAY}{ob_per_s:.0f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Ticks/s  {DKGRAY}{tik_per_s:.0f}{RESET}"
                f"{CLR}"
            )
            bstats = self._bot_stats.get(self._primary_bot_id or "", {}) if self._primary_bot_id else {}
            gen = int(bstats.get("generation", 0))
            run_id = str(bstats.get("run_id", "") or "")
            parent_run_id = str(bstats.get("parent_run_id", "") or "")
            scenario = str(bstats.get("scenario", "base") or "base")
            bought_contracts = float(bstats.get("buy_contracts", 0.0))
            run_short = run_id[:12] if run_id else "—"
            parent_short = parent_run_id[:12] if parent_run_id else "—"
            L.append(
                f" {AMBER}{probe[4]}{RESET}  {WHITE}Gen  {AMBER}{gen}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Scenario  {CYAN}{scenario}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Bought  {AMBER}{bought_contracts:.1f}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Run  {DKGRAY}{run_short}{RESET}"
                f"  {GRAY}│{RESET}  {WHITE}Parent  {DKGRAY}{parent_short}{RESET}"
                f"{CLR}"
            )

        # ── ORDERS ════════════════════════════════════════════════════
        total  = self._wins + self._losses
        wr_pct = f"{self._wins * 100 // total}%" if total > 0 else "---"
        wr_str = f"{self._wins}/{total} ({wr_pct})"
        open_c = AMBER if self._open_orders > 0 else GRAY
        orders_accent = _header_lights(frame)
        if _no_promoted:
            wr_str = "—"
            all_time_str = f"{WHITE}—{RESET}"
            fills_str = "—"
            open_str = "—"
        else:
            all_time_str = f"{at_c}{BOLD}${self._alltime_pnl:+.2f}{RESET}"
            fills_str = f"{self._total_fills}"
            open_str = f"{open_c}{BOLD}{self._open_orders}{RESET}"
        L.append(
            f" {ORANGE}{BOLD}══ ORDERS{RESET}"
            f"  {WHITE}{orders_accent}{RESET}"
            f"  {WHITE}Win Rate {RESET}{BOLD}{AMBER}{wr_str}{RESET}"
            f"  {GRAY}│{RESET}  {WHITE}All-Time {all_time_str}"
            f"  {GRAY}│{RESET}  {WHITE}Fills {fills_str}"
            f"  {GRAY}│{RESET}  {WHITE}Open Orders {open_str}"
            f"{CLR}"
        )
        L.append(f" {GRAY}{'─' * W}{RESET}{CLR}")
        if _no_promoted:
            L.append(f"  {GRAY}No promoted bot session yet — promoted fills will appear here after promotion{RESET}{CLR}")
        elif self._recent_fills:
            for fi in list(self._recent_fills)[:max_fills]:
                age_s   = int(time.time() - fi["ts"])
                age_str = f"{age_s}s ago" if age_s < 60 else f"{age_s // 60}m ago"
                side_c  = GREEN if fi["side"] == "yes" else RED
                mode_lbl = f"{DKGRAY}[PAPER]{RESET}" if self._dry_run else f"{DKGRAY}[LIVE]{RESET}"
                mkt_lbl  = _market_label(fi["ticker"], fi.get("source", ""))
                L.append(
                    f"  {mode_lbl}"
                    f"  {side_c}{BOLD}{fi['side'].upper():>3}{RESET}"
                    f"  {WHITE}{mkt_lbl:<12}{RESET}"
                    f"  {WHITE}@{fi['price']:>3}¢{RESET}"
                    f"  x{fi['count']:<5}"
                    f"  {GRAY}{age_str}{RESET}"
                    f"{CLR}"
                )
        elif self._dry_run:
            L.append(
                f"  {GRAY}SIM active — paper fills appear here when signals fire{RESET}{CLR}"
            )
        else:
            L.append(f"  {GRAY}No fills yet{RESET}{CLR}")

        # ── HISTORY ═══════════════════════════════════════════════════
        history_accent = _header_lights(frame)
        L.append(
            f" {ORANGE}{BOLD}══ HISTORY{RESET}"
            f"  {WHITE}{history_accent}{RESET}"
            f"  {WHITE}Uptime {BOLD}{WHITE}{up_str}{RESET}"
            f"{CLR}"
        )
        L.append(f" {GRAY}{'─' * W}{RESET}{CLR}")
        if _no_promoted:
            L.append(f"  {GRAY}No promoted bot session yet — promoted settlements will appear here after promotion{RESET}{CLR}")
        elif self._history:
            for h in list(self._history)[:max_hist]:
                tag_c   = GREEN if h["won"] else RED
                tag     = "WIN " if h["won"] else "LOSS"
                pnl_c   = GREEN if h["pnl"] >= 0 else RED
                mkt_lbl = _market_label(h["ticker"], h.get("source", ""))
                bid_lbl = f" {GRAY}({h.get('bot_id', '???')[:8]}){RESET}" if h.get("bot_id") != "default" else ""
                L.append(
                    f"  {tag_c}{BOLD}{tag}{RESET}"
                    f"  {WHITE}{mkt_lbl:<12}{RESET}"
                    f"  {pnl_c}{BOLD}${h['pnl']:+.2f}{RESET}"
                    f"  {h['side'].upper()}{bid_lbl}"
                    f"{CLR}"
                )
        else:
            L.append(f"  {GRAY}No settled trades yet{RESET}{CLR}")

        self._append_leaderboard(L, W, rotator)

        # Final rendering: home + clear-to-end keeps content top-anchored
        # without full-screen flashing.
        max_lines = max(24, min(80, rows - 1))
        lines_out = L[:max_lines]
        output = "\033[H" + "\n".join(lines_out) + "\033[J"
        # On Windows subprocess console, one huge write can crash; write in chunks to reduce risk.
        chunk_size = 2048
        try:
            for i in range(0, len(output), chunk_size):
                sys.stdout.write(output[i : i + chunk_size])
            sys.stdout.flush()
        except (OSError, UnicodeEncodeError):
            pass
