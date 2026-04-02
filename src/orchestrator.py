# Created by Oliver Meihls

# Argus Market Monitor - Module Loader / Orchestrator
#
# Coordinates all connectors, detectors, and alerts via a central
# Pub/Sub event bus.  Reads ``ARGUS_MODE`` once at boot to decide
# between **collector** (default — observe only) and **live** modes.
# This is the canonical orchestrator entrypoint; legacy variants have been removed.

import asyncio
import os
import signal
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.config import load_all_config, validate_secrets, get_secret, ZeusConfig
from .core.database import Database
from .core.logger import setup_logger, get_logger, uptime_seconds
from .core.bus import EventBus
from .core.events import (
    HeartbeatEvent,
    MinuteTickEvent,
    BarEvent,
    QuoteEvent,
    MetricEvent,
    SignalEvent,
    ExternalMetricEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SIGNALS,
    TOPIC_SIGNALS_RAW,
    TOPIC_OPTIONS_CHAINS,
    TOPIC_EXTERNAL_METRICS,
    TOPIC_SYSTEM_HEARTBEAT,
    TOPIC_SYSTEM_MINUTE_TICK,
)
from .core.status_tracker import ActivityStatusTracker
from .core.bar_builder import BarBuilder
from .core.persistence import PersistenceManager
from .core.feature_builder import FeatureBuilder
from .core.regime_detector import RegimeDetector
from .core.market_regime_detector import MarketRegimeDetector
from .core.gap_risk_tracker import GapRiskTracker
from .core.reddit_monitor import RedditMonitor
from .core.sentiment_collector import SentimentCollector
from .core.conditions_monitor import ConditionsMonitor
from .connectors.bybit_ws import BybitWebSocket
from .connectors.coinbase_ws import CoinbaseWebSocket
from .connectors.okx_ws_fallback import OkxWsFallback
from .connectors.deribit_client import DeribitClient
from .connectors.yahoo_client import YahooFinanceClient
from .connectors.alpaca_client import AlpacaDataClient
from .connectors.alpaca_options import AlpacaOptionsConnector, AlpacaOptionsConfig
from .connectors.tastytrade_options import TastytradeOptionsConnector, TastytradeOptionsConfig
from .connectors.public_client import PublicAPIClient, PublicAPIConfig
from .connectors.public_options import PublicOptionsConnector, PublicOptionsConfig
from .connectors.tastytrade_streamer import TastytradeStreamer
from .connectors.alphavantage_client import AlphaVantageClient
from .connectors.alphavantage_collector import AlphaVantageCollector
from .core.global_risk_flow_updater import GlobalRiskFlowUpdater
from .core.news_sentiment_updater import NewsSentimentUpdater, format_news_sentiment_telegram
from .core.greeks_cache import GreeksCache, enrich_snapshot_iv
from .core.iv_consensus import IVConsensusConfig, IVConsensusEngine
from .strategies.spread_generator import SpreadCandidateGenerator, SpreadGeneratorConfig
from .connectors.polymarket_gamma import PolymarketGammaClient
from .connectors.polymarket_clob import PolymarketCLOBClient
from .connectors.polymarket_watchlist import PolymarketWatchlistService
from .detectors.options_iv_detector import OptionsIVDetector
from .detectors.volatility_detector import VolatilityDetector
from .detectors.etf_options_detector import ETFOptionsDetector
from .alerts.telegram_bot import TelegramBot
from .analysis.daily_review import DailyReview
from .analysis.uniformity_monitor import run_uniformity_check
from .trading.paper_trader_farm import PaperTraderFarm
from .core.query_layer import QueryLayer
from .dashboard.web import ArgusWebDashboard
from .soak.guards import SoakGuardian
from .soak.tape import TapeRecorder
from .soak.resource_monitor import ResourceMonitor
from .soak.summary import build_soak_summary
from .core.liquid_etf_universe import get_liquid_etf_universe
from .agent.zeus import ZeusPolicyEngine, RuntimeMode
from .agent.delphi import DelphiToolRegistry
from .agent.runtime_controller import RuntimeController
from .agent.argus_orchestrator import ArgusOrchestrator as AgentOrchestrator

# Default options/IV underlyings: IBIT, BITO + liquid ETF universe
_DEFAULT_OPTIONS_SYMBOLS = sorted({"IBIT", "BITO"} | set(get_liquid_etf_universe()))


class CollectorModeViolation(RuntimeError):
    # Raised when trade-execution code is invoked in collector mode.


    pass

def _guard_collector_mode(mode: str):
    # Fail-fast if any module attempts trade execution in collector mode.
    if mode == "collector":
        raise CollectorModeViolation(
            "Trade execution attempted while ARGUS_MODE=collector. "
            "Set ARGUS_MODE=live to enable trading."
        )



class ArgusOrchestrator:
    # Main Argus orchestrator.
    #
    # Coordinates:
    # - Exchange WebSocket connections
    # - Data polling clients
    # - Opportunity detectors
    # - Alert dispatching
    
    def __init__(self, config_dir: str = "config"):
        # Initialize Argus.
        #
        # Args:
        # config_dir: Path to config directory
        # Load configuration
        self.config = load_all_config(config_dir)
        self.secrets = self.config.get('secrets', {})

        # ── Global mode (read once at boot) ─────────────
        self.mode: str = os.environ.get("ARGUS_MODE", "collector").lower()

        # Recent log lines ring buffer for dashboard (must be before logger setup)
        self._recent_logs: deque = deque(maxlen=200)

        # Setup logging (ARGUS_LOG_LEVEL env overrides config, e.g. for: python main.py --log-level DEBUG)
        log_level = os.environ.get('ARGUS_LOG_LEVEL') or self.config.get('system', {}).get('log_level', 'INFO')
        setup_logger('argus', level=log_level, ring_buffer=self._recent_logs)
        self.logger = get_logger('orchestrator')

        # ── Collector-mode banner ───────────────────────
        if self.mode == "collector":
            banner = (
                "\n"
                "╔══════════════════════════════════════════╗\n"
                "║  TRADING DISABLED  (COLLECTOR MODE)      ║\n"
                "║  Set ARGUS_MODE=live to enable trading.   ║\n"
                "╚══════════════════════════════════════════╝"
            )
            self.logger.warning(banner)
        else:
            self.logger.info(f"ARGUS_MODE = {self.mode}")

        # Validate secrets
        issues = validate_secrets(self.secrets)
        if issues:
            self.logger.warning("Configuration issues:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")

        # Initialize database
        db_path = Path(self.config.get('system', {}).get('database_path', 'data/argus.db'))
        self.db = Database(str(db_path))

        # Kalshi Sidecar DB (Read-only status monitoring)
        kalshi_db_path = Path('data/kalshi.db')
        self.kalshi_db = Database(str(kalshi_db_path))

        # ── Event bus + modules ─────────────────────────
        self.event_bus = EventBus()
        self.bar_builder: Optional[BarBuilder] = None
        self.persistence: Optional[PersistenceManager] = None
        self.query_layer: Optional[QueryLayer] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.market_regime_detector: Optional[MarketRegimeDetector] = None
        self._provider_names = [
            "bybit",
            "deribit",
            "yahoo",
            "alpaca",
            "binance",
            "okx",
            "coinglass",
            "coinbase",
            "ibit_options",
            "tastytrade_options",
            "public_options",
            "polymarket_gamma",
            "polymarket_clob",
            "alphavantage",
        ]
        self._detector_names = ["etf_options", "volatility_detector", "risk_flow"]
        self._activity_tracker = ActivityStatusTracker(
            provider_names=self._provider_names,
            detector_names=self._detector_names,
            boot_ts=time.time(),
        )

        # Polymarket connectors
        self.polymarket_gamma: Optional[PolymarketGammaClient] = None
        self.polymarket_clob: Optional[PolymarketCLOBClient] = None
        self.polymarket_watchlist: Optional[PolymarketWatchlistService] = None

        # Get symbols to monitor
        self.symbols = self.config.get('symbols', {}).get('monitored', [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'ARB/USDT:USDT', 'DOGE/USDT:USDT'
        ])
        
        # Components will be initialized in setup()
        self.bybit_ws: Optional[BybitWebSocket] = None
        self.coinbase_ws: Optional[CoinbaseWebSocket] = None
        # OKX WS fallback for truth-critical paths (activated when Coinbase silent > 5s)
        self._okx_fallback: Optional[Any] = None
        # ── Truth-feed health tracking ──────────────────────────
        # Source priority for truth-critical logic:
        #   1. Coinbase WS (primary) — US-regulated, sub-second ticks
        #   2. OKX WS (secondary fallback) — activated when Coinbase silent > _truth_fallback_activation_s
        #   3. Bybit WS — auxiliary analytics ONLY, never used for truth-critical decisions
        self._truth_feed_health: Dict[str, Dict] = {}  # asset -> {last_tick_mono, active_source, fallback_count}
        self._truth_fallback_activation_s: float = 5.0  # OKX fires after Coinbase silent this long
        self._truth_source_switch_log_ts: float = 0.0   # rate-limit source switch logs
        self.deribit_client: Optional[DeribitClient] = None
        self.yahoo_client: Optional[YahooFinanceClient] = None
        self.alpaca_client: Optional[AlpacaDataClient] = None
        self.alpaca_options: Optional[AlpacaOptionsConnector] = None
        self.tastytrade_options: Optional[TastytradeOptionsConnector] = None
        self.public_options: Optional[PublicOptionsConnector] = None
        self.alphavantage_client: Optional[AlphaVantageClient] = None
        self.av_collector: Optional[AlphaVantageCollector] = None
        self.global_risk_flow_updater: Optional[GlobalRiskFlowUpdater] = None
        self.news_sentiment_updater: Optional[NewsSentimentUpdater] = None
        self.spread_generator: Optional[SpreadCandidateGenerator] = None
        self._greeks_cache: GreeksCache = GreeksCache()
        self._iv_consensus = IVConsensusEngine(IVConsensusConfig(policy="prefer_public"))
        self._dxlink_streamer: Optional[TastytradeStreamer] = None
        self.telegram: Optional[TelegramBot] = None
        
        # Off-hours monitoring
        self.gap_risk_tracker: Optional[GapRiskTracker] = None
        self.reddit_monitor: Optional[RedditMonitor] = None
        
        # Conditions synthesis and daily review
        self.conditions_monitor: Optional[ConditionsMonitor] = None
        self.daily_review: Optional[DailyReview] = None
        
        # Paper trader farm (752 parallel traders)
        self.paper_trader_farm: Optional[PaperTraderFarm] = None
        self._recent_farm_entries: List[Any] = []
        self._recent_farm_exits: List[Any] = []
        self._farm_summary_last_run: float = time.time()
        
        self.detectors: Dict[str, Any] = {}
        
        self._running = False
        # Task-tracking invariant: every asyncio.create_task() in run()
        # MUST be appended to self._tasks so that stop() can cancel and
        # await all of them via asyncio.gather(*self._tasks).
        self._tasks: List[asyncio.Task] = []
        self._start_time = datetime.now(timezone.utc)
        self._last_price_snapshot: Dict[str, datetime] = {}
        self._last_health_check: Optional[datetime] = None
        self._research_last_run: Optional[datetime] = None
        self._research_last_symbol: Optional[str] = None
        self._research_last_entered: int = 0
        self._research_last_error: Optional[str] = None
        self._research_consecutive_errors: int = 0
        self._exit_monitor_last_run: Optional[datetime] = None
        self._research_promoted: bool = False
        self.research_config: Dict[str, Any] = self.config.get('research', {})
        self.research_enabled = self.research_config.get('enabled', False)
        self.research_alerts_enabled = self.research_config.get('alerts_enabled', False)
        self.research_daily_review_enabled = self.research_config.get('daily_review_enabled', False)
        self._deribit_traceback_ts = 0.0

        # Market session tracking
        self._market_was_open: bool = False
        self._last_market_open_date = None
        self._last_market_close_date = None
        self._today_opened: int = 0
        self._today_closed: int = 0
        self._today_expired: int = 0

        # Boot phase timing
        self._boot_phases: Dict[str, float] = {}
        self._boot_start = time.monotonic()

        # Cached status snapshots for fast dashboard/telegram responses
        self._status_snapshot: Dict[str, Any] = {}
        self._status_snapshot_ts: float = 0.0
        self._status_snapshot_lock = asyncio.Lock()
        self._status_snapshot_interval = int(
            self.config.get('monitoring', {}).get('status_snapshot_interval', 10)
        )
        self._zombies_snapshot: Dict[str, Any] = {'zombies': [], 'total': 0, 'report': 'No data yet'}
        self._zombies_snapshot_ts: float = 0.0
        self._zombies_snapshot_interval = int(
            self.config.get('monitoring', {}).get('zombies_snapshot_interval', 120)
        )

        # Dashboard
        dash_cfg = self.config.get('dashboard', {})
        self.dashboard: Optional[ArgusWebDashboard] = None
        if dash_cfg.get('enabled', True):
            self.dashboard = ArgusWebDashboard(
                host=dash_cfg.get('host', '127.0.0.1'),
                port=dash_cfg.get('port', 8777),
            )

        # ── Soak-test hardening components ────────────────
        soak_cfg = self.config.get('soak', {})
        db_path = str(self.config.get('system', {}).get('database_path', 'data/argus.db'))
        self.resource_monitor = ResourceMonitor(
            db_path=db_path,
            log_ring=self._recent_logs,
        )
        self.soak_guardian = SoakGuardian(
            config=soak_cfg.get('guards', {}),
            alert_callback=None,  # wired after telegram setup
        )
        tape_cfg = soak_cfg.get('tape', {})
        tape_symbols = set(tape_cfg.get('symbols', [])) or None
        self.tape_recorder = TapeRecorder(
            enabled=tape_cfg.get('enabled', False),
            symbols=tape_symbols,
            maxlen=int(tape_cfg.get('maxlen', 100_000)),
        )
        # Last component heartbeat timestamps (for guard checks)
        self._component_heartbeat_ts: Dict[str, float] = {}

        # Market hours config
        self._mh_cfg = self.config.get('market_hours', {})

        # AI agent stack (initialized in setup())
        self.ai_agent: Optional[AgentOrchestrator] = None

        self.logger.info("Argus Orchestrator initialized")
    
    def _phase(self, name: str):
        # Record a boot phase's elapsed time.
        elapsed = time.monotonic() - self._boot_start
        self._boot_phases[name] = round(elapsed, 2)
        self.logger.info(f"[BOOT] {name}: {elapsed:.2f}s")

    def _format_boot_phases(self) -> str:
        # Format boot phases for display.
        lines = []
        prev = 0.0
        for name, ts in self._boot_phases.items():
            delta = ts - prev
            lines.append(f"  {name}: {ts:.1f}s (delta {delta:.1f}s)")
            prev = ts
        return "\n".join(lines)

    def _note_detector_activity(self, detector: str, *, kind: str = "event") -> None:
        self._activity_tracker.record_detector_event(
            detector, event_ts=time.time(), kind=kind
        )

    def _note_detector_signal(self, event: SignalEvent) -> None:
        detector = getattr(event, "detector", None)
        if detector:
            self._activity_tracker.record_detector_signal(
                detector, event_ts=event.timestamp
            )

    def _note_provider_quote(self, event: QuoteEvent) -> None:
        provider = getattr(event, "source", None)
        if not provider:
            return
        self._activity_tracker.record_provider_event(
            provider,
            event_ts=event.event_ts,
            source_ts=event.source_ts or event.timestamp,
            kind="quote",
        )

    def _note_provider_bar(self, event: BarEvent) -> None:
        provider = getattr(event, "source", None)
        if not provider:
            return
        source_ts = (
            event.last_source_ts
            or event.first_source_ts
            or event.source_ts
            or event.timestamp
        )
        self._activity_tracker.record_provider_event(
            provider,
            event_ts=event.event_ts,
            source_ts=source_ts,
            kind="bar",
        )

    def _note_provider_metric(self, event: MetricEvent) -> None:
        provider = getattr(event, "source", None)
        if not provider:
            return
        self._activity_tracker.record_provider_event(
            provider,
            event_ts=event.event_ts,
            source_ts=event.source_ts or event.timestamp,
            kind="metric",
        )

    def _note_external_metric(self, event: ExternalMetricEvent) -> None:
        # Record activity for external metrics (e.g. risk flow).
        self._activity_tracker.record_detector_event(
            "risk_flow",
            event_ts=event.timestamp_ms / 1000.0,
            kind="metric",
        )

    def _wire_activity_tracking(self) -> None:
        self.event_bus.subscribe(TOPIC_MARKET_QUOTES, self._note_provider_quote)
        self.event_bus.subscribe(TOPIC_MARKET_BARS, self._note_provider_bar)
        self.event_bus.subscribe(TOPIC_MARKET_METRICS, self._note_provider_metric)
        self.event_bus.subscribe(TOPIC_EXTERNAL_METRICS, self._note_external_metric)
        self.event_bus.subscribe(TOPIC_SIGNALS, self._note_detector_signal)

    def _record_detector_activity(self, detector: str, event_ts: float, kind: str) -> None:
        if kind == "signal":
            self._activity_tracker.record_detector_signal(detector, event_ts=event_ts)
        else:
            self._activity_tracker.record_detector_event(
                detector, event_ts=event_ts, kind=kind
            )

    def _sync_provider_registry(self) -> None:
        providers = {
            "bybit": self.bybit_ws,
            "deribit": self.deribit_client,
            "yahoo": self.yahoo_client,
            "alpaca": self.alpaca_client,
            "binance": getattr(self, "binance_ws", None),
            "okx": getattr(self, "okx_client", None),
            "coinglass": getattr(self, "coinglass_client", None),
            "coinbase": getattr(self, "coinbase_client", None),
            "ibit_options": getattr(self, "ibit_options_client", None),
            "tastytrade_options": self.tastytrade_options,
            "public_options": self.public_options,
            "polymarket_gamma": self.polymarket_gamma,
            "polymarket_clob": self.polymarket_clob,
        }
        for name in self._provider_names:
            self._activity_tracker.register_provider(
                name, configured=providers.get(name) is not None
            )
        
        # Start background tasks for consolidated summaries
        self._tasks.append(asyncio.create_task(self._run_periodic_farm_summary()))

    async def setup(self) -> None:
        # Initialize all components with phase timing.
        self.logger.info("Setting up Argus components...")
        self._boot_start = time.monotonic()

        # Phase 1: Config (already done in __init__)
        self._phase("config_loaded")

        # Phase 2: Database (full schema created on connect)
        await self.db.connect()
        await self.kalshi_db.connect() # Sidecar
        self._phase("db_connected")

        # Phase 2b: Event bus modules
        loop = asyncio.get_running_loop()
        self.bar_builder = BarBuilder(self.event_bus)
        self.persistence = PersistenceManager(self.event_bus, self.db, loop)
        self.persistence.start()

        # Attach tape recorder to event bus (subscribes only if enabled)
        self.tape_recorder.attach(self.event_bus)

        # Phase 2c: Intelligence pipeline (downstream-only, safe in collector mode)
        self.feature_builder = FeatureBuilder(self.event_bus)
        self.regime_detector = RegimeDetector(self.event_bus)
        risk_basket = self.config.get("system", {}).get("risk_basket_symbols", [])
        self.market_regime_detector = MarketRegimeDetector(self.event_bus, risk_basket_symbols=risk_basket)
        self._wire_activity_tracking()
        self._phase("event_bus_wired")

        # Phase 3: Connectors (with event bus)
        await self._setup_connectors()
        self._sync_provider_registry()
        self._phase("connectors_init")

        # Phase 4: Telegram
        await self._setup_telegram()
        if self.telegram and self.av_collector:
            self.av_collector.set_telegram(self.telegram)
        self._phase("telegram_init")

        # Phase 5: Providers (gap risk, conditions, farm, review)
        # In collector mode, skip paper traders and exit monitors
        await self._setup_off_hours_monitoring()
        self._phase("providers_init")

        # Phase 6: Wire callbacks
        self._wire_telegram_callbacks()

        # Wire soak guardian alerts to telegram
        if self.telegram:
            self.soak_guardian._alert_cb = self._send_soak_alert

        # Phase 6b: AI agent stack (Zeus, Delphi, RuntimeController, AgentOrchestrator)
        try:
            zeus_config = ZeusConfig(
                monthly_budget_cap=15.0,
                governance_db_path=str(
                    Path(self.config.get('system', {}).get('database_path', 'data/argus.db')).parent
                    / "argus_governance.db"
                ),
            )
            zeus = ZeusPolicyEngine(zeus_config)
            delphi = DelphiToolRegistry(
                zeus=zeus,
                role_tool_allowlist={"Argus": {"*"}},
            )
            delphi.discover_tools()

            class _SimpleResourceHandle:
                gpu_enabled: bool = True

            runtime = RuntimeController(zeus=zeus, resource_manager=_SimpleResourceHandle())
            self.ai_agent = AgentOrchestrator(
                zeus=zeus,
                delphi=delphi,
                runtime=runtime,
                get_kalshi_summary=self._get_kalshi_summary,
                get_status=self._get_dashboard_system_status,
                get_pnl=self._get_pnl_summary,
                get_farm_status=self._get_farm_status,
            )
            # Wire up the chat bridge since agent is now ready
            if self.telegram:
                self.telegram.set_chat_callback(self._handle_ai_chat)
                
            self.logger.info("AI agent stack initialized (Zeus + Delphi + RuntimeController)")
        except Exception:
            self.logger.warning("AI agent stack failed to initialize — chat disabled", exc_info=True)
            self.ai_agent = None
        self._phase("ai_agent_init")

        # Phase 7: Detectors (with bus attachment)
        await self._setup_detectors()
        self._phase("detectors_init")

        # Phase 7b: Polymarket connectors (optional, fail-soft)
        await self._setup_polymarket()
        self._sync_provider_registry()
        self._phase("polymarket_init")

        # Phase 7c: Query layer (unified command interface)
        connectors_map: Dict[str, Any] = {
            "bybit": self.bybit_ws,
            "deribit": self.deribit_client,
            "yahoo": self.yahoo_client,
        }
        if self.polymarket_gamma:
            connectors_map["polymarket_gamma"] = self.polymarket_gamma
        if self.polymarket_clob:
            connectors_map["polymarket_clob"] = self.polymarket_clob
        self.query_layer = QueryLayer(
            bus=self.event_bus,
            db=self.db,
            detectors=self.detectors,
            connectors=connectors_map,
            bar_builder=self.bar_builder,
            persistence=self.persistence,
            feature_builder=self.feature_builder,
            regime_detector=self.regime_detector,
            provider_names=self._provider_names,
        )
        self._phase("query_layer_init")

        # Phase 7d: Start the event bus (all subscriptions registered)
        self.event_bus.start()
        self._phase("event_bus_started")

        # Phase 8: Zombie cleanup (skip full-table scan, use targeted query)
        await self._cleanup_zombie_positions()
        self._phase("zombie_cleanup")

        # Phase 9: Dashboard
        if self.dashboard:
            self.dashboard.set_callbacks(
                get_status=self._get_dashboard_system_status,
                get_pnl=self._get_pnl_summary,
                get_farm_status=self._get_farm_status,
                get_providers=self._get_provider_statuses,
                run_command=self._run_dashboard_command,
                get_recent_logs=self._get_recent_logs_text,
                get_soak_summary=self._get_soak_summary,
                export_tape=self._export_tape,
            )
            await self.dashboard.start()
            self.dashboard.set_boot_phases(self._format_boot_phases())
            self._phase("dashboard_started")

        # Total
        total = time.monotonic() - self._boot_start
        self._boot_phases["ready"] = round(total, 2)
        self.logger.info(f"[BOOT] READY in {total:.1f}s")
        self.logger.info(f"Boot phases:\n{self._format_boot_phases()}")

        # Startup notification
        if self.telegram:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern).strftime('%H:%M:%S %Z')
            farm_count = len(self.paper_trader_farm.trader_configs) if self.paper_trader_farm else 0
            dash_port = None
            if self.dashboard:
                dash_port = self.dashboard.port
            if dash_port is None:
                dash_port = self.config.get('dashboard', {}).get('port', 8777)
            startup_msg = (
                f"<b>Argus Online</b>\n"
                f"<i>{now_et}</i>\n\n"
                f"Detectors: {len(self.detectors)}\n"
                f"Farm: {farm_count:,} configs\n"
                f"Boot: {total:.1f}s\n"
                f"Dashboard: http://127.0.0.1:{dash_port}"
            )
            await self.telegram.send_message(startup_msg)

        self.logger.info("Setup complete!")
    
    async def _setup_connectors(self) -> None:
        # Initialize exchange connectors (with event bus wiring).
        av_cfg = self.config.get('exchanges', {}).get('alphavantage', {})
        if av_cfg.get('enabled', False):
            av_api_key = get_secret(self.secrets, 'alphavantage', 'api_key')
            if av_api_key:
                # No retries on rate limit so we don't burn daily quota (25/day free tier)
                self.alphavantage_client = AlphaVantageClient(
                    api_key=av_api_key,
                    max_retries=0,
                    call_interval_seconds=15.0,
                )
                self.logger.info("Alpha Vantage client configured (15s interval, no retries)")
                
                # Continuous Collector
                self.av_collector = AlphaVantageCollector(
                    av_client=self.alphavantage_client,
                    db=self.db,
                    bus=self.event_bus,
                    config=self.config
                )
            else:
                self.logger.warning("Alpha Vantage enabled but API key not found in secrets.yaml")

        # GlobalRiskFlow Updater
        self.global_risk_flow_updater = GlobalRiskFlowUpdater(
            bus=self.event_bus,
            db=self.db,
            config=self.config,
        )
        self.news_sentiment_updater = NewsSentimentUpdater(
            bus=self.event_bus,
            config=self.config,
        )

        # Bybit WebSocket (public - no auth needed)
        bybit_symbols = [s for s in self.symbols]
        self.bybit_ws = BybitWebSocket(
            symbols=bybit_symbols,
            on_ticker=self._on_bybit_ticker,
            on_funding=self._on_funding_update,
            event_bus=self.event_bus,
        )
        self.logger.info(f"Bybit WS configured for {len(bybit_symbols)} symbols")

        # Coinbase WebSocket (public/ticker)
        coinbase_symbols = ["BTC/USDT", "ETH/USDT"]
        self.coinbase_ws = CoinbaseWebSocket(
            symbols=coinbase_symbols,
            on_ticker=self._on_coinbase_ticker,
            event_bus=self.event_bus,
        )
        self.logger.info(f"Coinbase WS configured for {len(coinbase_symbols)} symbols")

        # OKX WebSocket fallback (public - no auth, fires only when Coinbase silent > 5s)
        # Source priority: Coinbase (primary) > OKX (fallback) > Bybit (analytics-only)
        coinbase_assets = ["BTC", "ETH"]  # assets covered by Coinbase truth feed
        for asset in coinbase_assets:
            self._truth_feed_health[asset] = {
                "last_tick_mono": 0.0,
                "active_source": "coinbase",
                "fallback_count": 0,
            }

        async def _on_okx_orchestrator_ticker(data: Dict) -> None:
            # OKX fallback handler — only publishes when Coinbase is silent.
            symbol = data.get("symbol", "")
            # Map OKX instId to asset
            asset = "BTC" if "BTC" in symbol else "ETH" if "ETH" in symbol else None
            if not asset:
                return
            health = self._truth_feed_health.get(asset)
            if not health:
                return
            now_mono = time.monotonic()
            coinbase_age = now_mono - health["last_tick_mono"]
            if health["last_tick_mono"] > 0 and coinbase_age < self._truth_fallback_activation_s:
                return  # Coinbase is alive, suppress OKX
            # Coinbase silent — OKX is now truth source
            health["active_source"] = "okx"
            health["fallback_count"] += 1
            # Rate-limit source switch log (once per 30s)
            if (now_mono - self._truth_source_switch_log_ts) > 30.0:
                self.logger.info(
                    "Truth feed switch: %s -> OKX fallback (Coinbase silent %.1fs)",
                    asset, coinbase_age,
                )
                self._truth_source_switch_log_ts = now_mono
            data["exchange"] = "okx_fallback"
            await self._maybe_log_price_snapshot(
                exchange="okx_fallback",
                symbol=data.get("symbol"),
                price=data.get("last_price"),
                volume=data.get("volume_24h"),
            )

        self._okx_fallback = OkxWsFallback(
            inst_ids=["BTC-USDT", "ETH-USDT"],
            on_ticker=_on_okx_orchestrator_ticker,
        )
        self.logger.info("OKX WS fallback configured (secondary truth feed)")

        # Deribit REST client (public - no auth needed)
        # Use mainnet for real data
        self.deribit_client = DeribitClient(testnet=False, event_bus=self.event_bus)
        self.logger.info("Deribit client configured (mainnet)")

        # Yahoo Finance for equity/ETF quotes -> 1m bars via BarBuilder
        yahoo_cfg = self.config.get('exchanges', {}).get('yahoo', {})
        if yahoo_cfg.get('enabled', True):
            yahoo_symbols = yahoo_cfg.get('symbols', ['IBIT', 'BITO', 'SPY', 'QQQ', 'NVDA'])
            self.yahoo_client = YahooFinanceClient(
                symbols=yahoo_symbols,
                on_update=self._on_yahoo_update,
                event_bus=self.event_bus,
            )
            self.logger.info("Yahoo Finance client configured for %s", yahoo_symbols)
        else:
            self.logger.info("Yahoo Finance client disabled (set exchanges.yahoo.enabled=true)")

        # Alpaca Data Client for IBIT/BITO bars (runs in ALL modes - data only, no execution)
        alpaca_cfg = self.config.get('exchanges', {}).get('alpaca', {})
        if alpaca_cfg.get('enabled', False):
            alpaca_api_key = get_secret(self.secrets, 'alpaca', 'api_key')
            alpaca_api_secret = get_secret(self.secrets, 'alpaca', 'api_secret')
            if alpaca_api_key and alpaca_api_secret:
                self.alpaca_client = AlpacaDataClient(
                    api_key=alpaca_api_key,
                    api_secret=alpaca_api_secret,
                    symbols=alpaca_cfg.get('symbols', ['IBIT', 'BITO']),
                    event_bus=self.event_bus,
                    db=self.db,  # For restart dedupe initialization
                    poll_interval=int(alpaca_cfg.get('poll_interval_seconds', 60)),
                    overlap_seconds=int(alpaca_cfg.get('overlap_seconds', 120)),
                )
                self.logger.info(
                    "Alpaca Data client configured for %s (fixed poll_interval=%ds, overlap=%ds)",
                    alpaca_cfg.get('symbols', ['IBIT', 'BITO']),
                    alpaca_cfg.get('poll_interval_seconds', 60),
                    alpaca_cfg.get('overlap_seconds', 120),
                )
                
                # Alpaca Options Connector (Phase 3B)
                options_cfg = alpaca_cfg.get('options', {})
                if options_cfg.get('enabled', False):
                    self.alpaca_options = AlpacaOptionsConnector(
                        config=AlpacaOptionsConfig(
                            api_key=alpaca_api_key,
                            api_secret=alpaca_api_secret,
                            cache_ttl_seconds=int(options_cfg.get('poll_interval_seconds', 60)),
                        )
                    )
                    self._alpaca_options_symbols = options_cfg.get('symbols', _DEFAULT_OPTIONS_SYMBOLS)
                    self._alpaca_options_poll_interval = int(options_cfg.get('poll_interval_seconds', 60))
                    self._alpaca_options_min_dte = int(options_cfg.get('min_dte', 7))
                    self._alpaca_options_max_dte = int(options_cfg.get('max_dte', 21))
                    
                    # SpreadCandidateGenerator (subscribes to options.chains)
                    self.spread_generator = SpreadCandidateGenerator(
                        strategy_id="PUT_SPREAD_V1",
                        config=SpreadGeneratorConfig(
                            min_dte=self._alpaca_options_min_dte,
                            max_dte=self._alpaca_options_max_dte,
                        ),
                        on_signal=self._on_spread_signal,
                    )
                    # Subscribe generator to chain snapshots
                    self.event_bus.subscribe(
                        TOPIC_OPTIONS_CHAINS,
                        self.spread_generator.on_chain_snapshot,
                    )
                    self.logger.info(
                        "Alpaca Options connector + SpreadGenerator configured for %s (poll=%ds, DTE=%d-%d)",
                        self._alpaca_options_symbols,
                        self._alpaca_options_poll_interval,
                        self._alpaca_options_min_dte,
                        self._alpaca_options_max_dte,
                    )
                else:
                    self.logger.info("Alpaca Options disabled (set alpaca.options.enabled=true)")
            else:
                self.logger.warning("Alpaca enabled but API keys not found in secrets.yaml")
        else:
            self.logger.info("Alpaca Data client disabled (set exchanges.alpaca.enabled=true to activate)")

        # ── Tastytrade Options Snapshot Connector ────────────────────────
        tt_cfg = self.config.get('tastytrade', {})
        tt_sampling = tt_cfg.get('snapshot_sampling', {})
        if tt_sampling.get('enabled', False):
            tt_secrets = self.secrets.get('tastytrade', {})
            tt_oauth = self.secrets.get('tastytrade_oauth2', {})
            tt_username = tt_secrets.get('username', '')
            tt_password = tt_secrets.get('password', '')
            oauth_client_id = tt_oauth.get('client_id', '')
            oauth_client_secret = tt_oauth.get('client_secret', '')
            oauth_refresh_token = tt_oauth.get('refresh_token', '')
            has_oauth = (
                oauth_client_id and oauth_client_secret and oauth_refresh_token
                and not oauth_client_id.startswith('PASTE_')
            )
            has_session = tt_username and tt_password and not tt_username.startswith('PASTE_')
            if has_oauth or has_session:
                retry_cfg = tt_cfg.get('retries', {})
                alpaca_opts_symbols = getattr(self, '_alpaca_options_symbols', [])
                tt_underlyings = tt_cfg.get('underlyings', [])
                if alpaca_opts_symbols:
                    tt_options_symbols = [s for s in tt_underlyings if s in alpaca_opts_symbols]
                    if not tt_options_symbols:
                        tt_options_symbols = tt_underlyings[:4]
                else:
                    tt_options_symbols = tt_underlyings

                tt_min_dte = getattr(self, '_alpaca_options_min_dte', 7)
                tt_max_dte = getattr(self, '_alpaca_options_max_dte', 21)
                tt_poll_interval = getattr(self, '_alpaca_options_poll_interval', 60)

                try:
                    self.tastytrade_options = TastytradeOptionsConnector(
                        config=TastytradeOptionsConfig(
                            username=tt_username,
                            password=tt_password,
                            environment=tt_cfg.get('environment', 'live'),
                            timeout_seconds=tt_cfg.get('timeout_seconds', 20),
                            max_attempts=retry_cfg.get('max_attempts', 3),
                            backoff_seconds=retry_cfg.get('backoff_seconds', 1.0),
                            backoff_multiplier=retry_cfg.get('backoff_multiplier', 2.0),
                            symbols=tt_options_symbols,
                            min_dte=tt_min_dte,
                            max_dte=tt_max_dte,
                            poll_interval_seconds=tt_poll_interval,
                            oauth_client_id=oauth_client_id,
                            oauth_client_secret=oauth_client_secret,
                            oauth_refresh_token=oauth_refresh_token,
                        )
                    )
                    self._tastytrade_options_symbols = tt_options_symbols
                    self._tastytrade_options_poll_interval = tt_poll_interval
                    self._tastytrade_options_min_dte = tt_min_dte
                    self._tastytrade_options_max_dte = tt_max_dte
                    self.logger.info(
                        "Tastytrade Options snapshot connector configured for %s (poll=%ds, DTE=%d-%d)",
                        tt_options_symbols, tt_poll_interval, tt_min_dte, tt_max_dte,
                    )
                except Exception as exc:
                    self.logger.error(
                        "Failed to initialize Tastytrade Options connector: %s", exc
                    )
                    self.tastytrade_options = None
            else:
                self.logger.info(
                    "Tastytrade Options disabled (need tastytrade_oauth2 client_id, client_secret, refresh_token "
                    "or tastytrade username/password in secrets.yaml)"
                )
        else:
            self.logger.info("Tastytrade snapshot sampling disabled (set tastytrade.snapshot_sampling.enabled=true)")

        # ── Public.com Options Snapshot Connector (hybrid chain structure + Public greeks) ──
        public_opts_cfg = self.config.get('public_options', {})
        if public_opts_cfg.get('enabled', False):
            public_secret_cfg = self.secrets.get('public', {})
            public_api_secret = public_secret_cfg.get('api_secret', '')
            public_cfg = self.config.get('public', {})
            public_account_id = str(
                public_secret_cfg.get('account_id')
                or public_cfg.get('account_id')
                or ''
            )

            if not public_api_secret or str(public_api_secret).startswith('PASTE_'):
                raise ValueError("public_options.enabled=true requires secrets.public.api_secret")
            public_account_id = (public_account_id or "").strip()
            if not public_account_id:
                raise ValueError(
                    "public_options.enabled=true requires public.account_id (set in config or secrets). "
                    "The Public API does not expose an accounts listing endpoint."
                )

            structure_connector = self.alpaca_options or self.tastytrade_options
            if structure_connector is None:
                self.logger.warning(
                    "Public options enabled but no structure connector is available; "
                    "enable exchanges.alpaca.options or tastytrade.snapshot_sampling"
                )
            else:
                structure_provider = 'alpaca' if self.alpaca_options else 'tastytrade'
                self._public_options_symbols = public_opts_cfg.get('symbols', _DEFAULT_OPTIONS_SYMBOLS)
                self._public_options_poll_interval = int(public_opts_cfg.get('poll_interval_seconds', 60))
                self._public_options_min_dte = int(public_opts_cfg.get('min_dte', 7))
                self._public_options_max_dte = int(public_opts_cfg.get('max_dte', 21))

                public_client = PublicAPIClient(
                    PublicAPIConfig(
                        api_secret=public_api_secret,
                        account_id=public_account_id,
                        base_url=public_cfg.get('base_url', 'https://api.public.com'),
                        timeout_seconds=float(public_cfg.get('timeout_seconds', 20.0)),
                        rate_limit_rps=int(public_cfg.get('rate_limit_rps', 10)),
                    )
                )
                self.public_options = PublicOptionsConnector(
                    client=public_client,
                    structure_connector=structure_connector,
                    config=PublicOptionsConfig(
                        symbols=self._public_options_symbols,
                        poll_interval_seconds=self._public_options_poll_interval,
                        min_dte=self._public_options_min_dte,
                        max_dte=self._public_options_max_dte,
                    ),
                )
                self.logger.info(
                    "Public options snapshot connector configured for %s (poll=%ds, DTE=%d-%d, structure=%s)",
                    self._public_options_symbols,
                    self._public_options_poll_interval,
                    self._public_options_min_dte,
                    self._public_options_max_dte,
                    structure_provider,
                )
        else:
            self.logger.info("Public options snapshots disabled (set public_options.enabled=true)")

        # SpreadCandidateGenerator: subscribe to options.chains when we have any options
        # chain source (Tastytrade or Public), so we don't require Alpaca options.
        if self.spread_generator is None and (self.tastytrade_options or self.public_options):
            min_dte = getattr(self, '_tastytrade_options_min_dte', None) or getattr(
                self, '_public_options_min_dte', 7
            )
            max_dte = getattr(self, '_tastytrade_options_max_dte', None) or getattr(
                self, '_public_options_max_dte', 21
            )
            self.spread_generator = SpreadCandidateGenerator(
                strategy_id="PUT_SPREAD_V1",
                config=SpreadGeneratorConfig(min_dte=min_dte, max_dte=max_dte),
                on_signal=self._on_spread_signal,
            )
            self.event_bus.subscribe(TOPIC_OPTIONS_CHAINS, self.spread_generator.on_chain_snapshot)
            self.logger.info(
                "SpreadCandidateGenerator configured from options chain source (DTE=%d-%d)",
                min_dte, max_dte,
            )

    async def _setup_polymarket(self) -> None:
        # Initialize Polymarket connectors (optional, fail-soft).
        pm_cfg = self.config.get('polymarket', {})
        if not pm_cfg.get('enabled', False):
            self.logger.info("Polymarket integration disabled (set polymarket.enabled=true to activate)")
            return

        try:
            self.polymarket_gamma = PolymarketGammaClient(
                event_bus=self.event_bus,
                poll_interval=pm_cfg.get('gamma_poll_interval', 60),
            )
            await self.polymarket_gamma.start()

            self.polymarket_clob = PolymarketCLOBClient(
                event_bus=self.event_bus,
                poll_interval=pm_cfg.get('clob_poll_interval', 30),
            )
            await self.polymarket_clob.start()

            self.polymarket_watchlist = PolymarketWatchlistService(
                gamma_client=self.polymarket_gamma,
                clob_client=self.polymarket_clob,
                sync_interval=pm_cfg.get('watchlist_sync_interval', 300),
                max_watchlist=pm_cfg.get('max_watchlist', 50),
                min_volume=pm_cfg.get('min_volume', 10_000),
                keywords=pm_cfg.get('keywords', []),
            )
            await self.polymarket_watchlist.start()
            self.logger.info("Polymarket integration initialised (Gamma + CLOB + Watchlist)")
        except Exception:
            self.logger.exception("Polymarket setup failed (non-fatal, continuing)")
            self.polymarket_gamma = None
            self.polymarket_clob = None
            self.polymarket_watchlist = None

    async def _setup_detectors(self) -> None:
        # Initialize all detectors and attach to event bus.
        thresholds = self.config.get('thresholds', {})
        is_collector = self.mode == "collector"

        def _register(detector) -> None:
            detector.set_activity_callback(self._record_detector_activity)
            self._activity_tracker.register_detector(detector.name)

        # Options IV detector (BTC options on Deribit - for research)
        iv_config = thresholds.get('options_iv', {})
        if iv_config.get('enabled', True):
            self.detectors['options_iv'] = OptionsIVDetector(iv_config, self.db)
            self.detectors['options_iv'].attach_bus(self.event_bus)
            _register(self.detectors['options_iv'])

        # Volatility detector
        vol_config = thresholds.get('volatility', {})
        if vol_config.get('enabled', True):
            self.detectors['volatility'] = VolatilityDetector(vol_config, self.db)
            self.detectors['volatility'].attach_bus(self.event_bus)
            _register(self.detectors['volatility'])

        # ETF Options Detectors (IBIT, BITO, SPY, QQQ, IWM)
        etf_symbols = ['IBIT', 'BITO', 'SPY', 'QQQ', 'IWM']
        for sym in etf_symbols:
            cfg_key = sym.lower()
            default_vol_proxy = 'BTC' if sym in ['IBIT', 'BITO'] else sym
            
            # Default thresholds vary by asset type
            if sym in ['IBIT', 'BITO']:
                def_iv_thresh = 25
                def_drop = -0.5
            else:
                def_iv_thresh = 15  # Equities have lower IV regimes
                def_drop = -1.5     # Equities move less than crypto ETFs
                
            cfg = thresholds.get(cfg_key, {
                'enabled': True,
                'vol_iv_threshold': def_iv_thresh,
                'drop_threshold': def_drop,
                'combined_score_threshold': 1.0,
                'cooldown_hours': 3,
                'vol_proxy': default_vol_proxy,
            })
            
            if cfg.get('enabled', True):
                name = f"etf_{cfg_key}"
                self.detectors[name] = ETFOptionsDetector(
                    cfg, self.db, symbol=sym, iv_consensus=self._iv_consensus
                )
                self.detectors[name].attach_bus(self.event_bus)
                _register(self.detectors[name])
                
                # Wire up farm if available and NOT in collector mode
                if self.paper_trader_farm and not is_collector:
                    self.detectors[name].set_paper_trader_farm(self.paper_trader_farm)
                if self.research_enabled or is_collector:
                    self.detectors[name].paper_trading_enabled = False

        if is_collector:
            self.logger.info("Collector mode: paper trading DISABLED for all detectors")

        self.logger.info(f"Initialized {len(self.detectors)} detectors (bus-attached)")
    
    async def _setup_telegram(self) -> None:
        # Initialize Telegram bot.
        bot_token = get_secret(self.secrets, 'telegram', 'bot_token')
        chat_id = get_secret(self.secrets, 'telegram', 'chat_id')
        
        if bot_token and chat_id and not bot_token.startswith('PASTE_'):
            self.telegram = TelegramBot(
                bot_token=bot_token,
                chat_id=chat_id,
            )
            
            # Test connection
            if await self.telegram.test_connection():
                self.logger.info("Telegram bot connected successfully")
            else:
                self.logger.error("Telegram connection failed")
                self.telegram = None
        else:
            self.logger.warning("Telegram not configured - alerts disabled")
    
    async def _setup_off_hours_monitoring(self) -> None:
        # Initialize gap risk tracker, conditions monitor, and daily review.
        thresholds = self.config.get('thresholds', {})
        
        # Gap Risk Tracker
        gap_config = thresholds.get('gap_risk', {})
        if gap_config.get('enabled', True):
            self.gap_risk_tracker = GapRiskTracker(self.db, gap_config)
            await self.gap_risk_tracker.initialize()
            self.logger.info("Gap Risk Tracker initialized")
        
        # Reddit Monitor (only if API keys configured)
        reddit_secrets = self.secrets.get('reddit', {})
        reddit_config = thresholds.get('reddit_sentiment', {})
        
        client_id = reddit_secrets.get('client_id', '')
        client_secret = reddit_secrets.get('client_secret', '')
        
        if client_id and client_secret and reddit_config.get('enabled', True):
            self.reddit_monitor = RedditMonitor(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=reddit_secrets.get('user_agent', 'Argus/1.0'),
            )
            self.logger.info("Reddit Monitor initialized")
        else:
            self.logger.info("Reddit Monitor not configured - sentiment tracking disabled")
        
        # Conditions Monitor (synthesis layer)
        conditions_config = thresholds.get('conditions_monitor', {})
        self.conditions_monitor = ConditionsMonitor(
            config=conditions_config,
            on_alert=self._on_conditions_alert,
        )
        
        # Wire up data sources to conditions monitor
        self.conditions_monitor.set_data_sources(
            get_btc_iv=self._get_btc_iv,
            get_funding=self._get_btc_funding,
            get_btc_price=self._get_btc_price,
            get_risk_flow=self._get_current_risk_flow,
        )
        self.logger.info("Conditions Monitor initialized")
        
        # Daily Review (4 PM summary)
        self.daily_review = DailyReview(
            starting_balance=5000.0,
            on_send=self._send_daily_review,
        )
        
        # Paper Trader Farm with guardrails from config
        # Skip in collector mode to avoid millions of trader configs and GPU uploads
        if self.mode == "collector":
            self.paper_trader_farm = None
            self.logger.info(
                "PaperTraderFarm initialization SKIPPED (ARGUS_MODE=collector)"
            )
        else:
            farm_cfg = self.config.get('farm', {})
            self.paper_trader_farm = PaperTraderFarm(
                db=self.db,
                full_coverage=True,
                starting_balance=float(farm_cfg.get('default_starting_equity', 5000.0)),
                max_traders=int(farm_cfg.get('max_traders', 2_000_000)),
                max_open_positions_total=int(farm_cfg.get('max_open_positions_total', 500_000)),
                max_trades_per_minute=int(farm_cfg.get('max_trades_per_minute', 10_000)),
            )
            await self.paper_trader_farm.initialize()

            # Wire up data sources to paper trader farm
            self.paper_trader_farm.set_data_sources(
                get_conditions=self.conditions_monitor.get_current_conditions,
            )
            # Wire Telegram callback for runaway safety alerts (using separate safety callback)
            self.paper_trader_farm.set_telegram_alert_callback(self._send_paper_notification)
            self.logger.info(f"Paper Trader Farm initialized with {len(self.paper_trader_farm.trader_configs):,} traders")

        # Wire up data sources to daily review (after farm is ready, if available)
        self.daily_review.set_data_sources(
            get_conditions=self.conditions_monitor.get_current_conditions,
            get_positions=(
                self.paper_trader_farm.get_positions_for_review
                if self.paper_trader_farm else None
            ),
            get_trade_stats=(
                self.paper_trader_farm.get_trade_activity_summary
                if self.paper_trader_farm else None
            ),
            get_gap_risk=self.gap_risk_tracker.get_status if self.gap_risk_tracker else None,
        )
        self.logger.info("Daily Review initialized")
        
    def _wire_telegram_callbacks(self) -> None:
        # Wire up Telegram two-way callbacks once dependencies are ready.
        if not self.telegram:
            return
        if not self.conditions_monitor:
            return
        self.telegram.set_callbacks(
            get_conditions=self._get_status_summary,
            get_pnl=self._get_pnl_summary,
            get_positions=self._get_positions_summary,
            get_farm_status=self._get_farm_status,
            get_signal_status=self._get_signal_status,
            get_research_status=self._get_research_status,
            get_dashboard=self._get_dashboard,
            get_zombies=self._get_zombies,
            get_followed=self._get_followed_traders,
            get_sentiment=self._get_sentiment_summary,
            set_mode=self._handle_telegram_mode_switch,
            get_kalshi_summary=self._get_kalshi_summary,
        )

        # Wire AI chat bridge & data perceptual tools
        if self.ai_agent is not None:
            self.telegram.set_chat_callback(self._handle_ai_chat)
            self.ai_agent.set_data_sources(
                get_status=self._get_status_summary,
                get_pnl=self._get_pnl_summary,
                get_positions=self._get_positions_summary,
                get_farm_status=self._get_farm_status,
                get_kalshi_summary=self._get_kalshi_summary,
            )

        # Wire up Soak Guardian alerts (filtered)
        if hasattr(self, 'soak_guardian'):
            self.soak_guardian._alert_cb = self._handle_soak_alert
    
    async def _handle_ai_chat(self, message: str) -> str:
        # Route a Telegram chat message through the AI agent with safety guards.
        if self.ai_agent is None:
            return "AI agent is not initialized. Check startup logs."

        # Mode awareness: if Zeus is in DATA_ONLY, the GPU/LLM is down
        try:
            current_mode = self.ai_agent.zeus.current_mode
        except Exception:
            current_mode = None

        if current_mode == RuntimeMode.DATA_ONLY:
            # Relax block for mode-switch intent
            mode_keywords = {"active", "gaming", "switch", "mode", "turn on", "wake up"}
            lower = message.lower()
            if not any(kw in lower for kw in mode_keywords):
                return (
                    "AI is currently hibernating (Gaming/DATA_ONLY mode). "
                    "GPU resources are reserved for other tasks. "
                    "Switch to ACTIVE mode to re-enable chat."
                )

        # Inject Live Context so Argus has "peripheral vision" on the system
        try:
            status = await self._get_status_summary()
            regime = status.get("warmth_label", status.get("regime_name", "Unknown"))
            score = status.get("score", status.get("btc_score", 0.0))
            data_fresh = "All Green" if all(v.get("is_fresh", True) for v in status.get("data_status", {}).values()) else "Warnings"
            
            context_snippet = (
                f"\n[SYSTEM STATUS]: Regime={regime}, BTC Score={score:.2f}, Data Health={data_fresh}. "
                "Use this to answer naturally."
            )
            message = f"{context_snippet}\n\n{message}"
        except Exception as e:
            self.logger.warning(f"AI context injection failed: {e}")

        try:
            return await self.ai_agent.chat(message)
        except Exception as exc:
            self.logger.error("AI chat failed: %s", exc, exc_info=True)
            return "Brain offline. Check Ollama status."

    async def _handle_telegram_mode_switch(self, target: str) -> str:
        # Bridge Telegram /mode command to the AI agent's governor.
        if self.ai_agent is None:
            return "AI agent not initialized."
        
        # Explicit request text for the agent's classifier
        intent_request = f"switch to {target} mode"
        try:
            return await self.ai_agent.chat(intent_request)
        except Exception as e:
            self.logger.error(f"Telegram mode switch failed: {e}")
            return f"Failed to switch mode: {e}"

    async def _on_conditions_alert(self, snapshot) -> None:
        # Handle conditions threshold crossing alert.
        if not self.telegram:
            return
        if self.research_enabled and not self.research_alerts_enabled:
            return

        details = {
            'BTC IV': f"{snapshot.btc_iv:.0f}% ({snapshot.iv_signal})",
            'Funding': f"{snapshot.funding_rate:+.3f}% ({snapshot.funding_signal})",
            'BTC': f"{snapshot.btc_change_24h:+.1f}% ({snapshot.momentum_signal})",
            'Market': "🟢 OPEN" if snapshot.market_open else "🔴 CLOSED",
        }
        
        await self.telegram.send_conditions_alert(
            score=snapshot.score,
            label=snapshot.label,
            details=details,
            implication=snapshot.implication,
        )
    
    async def _send_daily_review(self, message: str) -> None:
        # Send daily review via Telegram.
        if self.telegram and (not self.research_enabled or self.research_daily_review_enabled):
            await self.telegram.send_message(message)
    
    async def _get_btc_iv(self) -> Optional[Dict]:
        # Get current BTC IV from Deribit.
        if self.deribit_client:
            try:
                return await self.deribit_client.get_atm_iv('BTC')
            except Exception:
                pass

        return None
    
    async def _get_btc_funding(self) -> Optional[Dict]:
        # Get current BTC funding rate from Bybit.
        if self.bybit_ws:
            rate = self.bybit_ws.get_funding_rate('BTCUSDT')
            if rate is not None:
                return {'rate': rate}
        return None
    
    async def _get_btc_price(self) -> Optional[Dict]:
        # Get current BTC price.
        if self.bybit_ws:
            ticker = self.bybit_ws.get_ticker('BTCUSDT')
            if ticker:
                change_5d_pct = 0.0
                if self.db:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
                    past = await self.db.get_price_at_or_before(
                        exchange='bybit',
                        asset='BTCUSDT',
                        price_type='spot',
                        cutoff_timestamp=cutoff,
                    )
                    past_price = past.get('price') if past else None
                    current_price = ticker.get('last_price', 0)
                    if past_price and current_price:
                        change_5d_pct = ((current_price - past_price) / past_price) * 100
                return {
                    'price': ticker.get('last_price', 0),
                    'change_24h_pct': ticker.get('price_change_24h', 0),
                    'change_5d_pct': change_5d_pct,
                }
        return None
    
    async def _get_pnl_summary(self) -> Dict:
        # Get P&L summary for Telegram /pnl command.
        cached = self._get_snapshot_section('pnl')
        if cached:
            return cached
        await self._refresh_status_snapshot(force=True)
        return self._status_snapshot.get('pnl', {})
    
    async def _get_positions_summary(self) -> List[Dict]:
        # Get positions summary for Telegram /positions command.
        cached = self._get_snapshot_section('positions')
        if cached is not None:
            return cached
        await self._refresh_status_snapshot(force=True)
        return self._status_snapshot.get('positions', [])

    async def _get_status_summary(self) -> Dict[str, Any]:
        # Get conditions plus data freshness for Telegram /status command.
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        data_status = self._get_snapshot_section('data_status')
        if data_status is None:
            await self._refresh_status_snapshot(force=True)
            data_status = self._status_snapshot.get('data_status', {})
        conditions['data_status'] = data_status
        return conditions

    async def _get_farm_status(self) -> Dict[str, Any]:
        # Get paper trader farm status summary for Telegram.
        cached = self._get_snapshot_section('farm')
        if cached:
            return cached
        await self._refresh_status_snapshot(force=True)
        return self._status_snapshot.get('farm', {})

    async def _get_signal_status(self) -> Dict[str, Any]:
        # Get signal checklist for all active ETF detectors.
        status: Dict[str, Any] = {}
        for key, detector in self.detectors.items():
            if hasattr(detector, 'get_signal_checklist'):
                status[detector.symbol] = detector.get_signal_checklist()
        return status

    async def _get_research_status(self) -> Dict[str, Any]:
        # Get research mode telemetry for Telegram.
        if not self.paper_trader_farm:
            return {}
        aggregate = self.paper_trader_farm.get_aggregate_pnl()
        status = self.paper_trader_farm.get_status_summary()
        data_ready = False
        for detector_name, detector in self.detectors.items():
            if detector_name.startswith('etf_') and hasattr(detector, 'get_signal_checklist'):
                checklist = detector.get_signal_checklist()
                data_ready = data_ready or (checklist.get('has_proxy_iv') and checklist.get('has_symbol_data'))
        return {
            'research_enabled': self.research_enabled,
            'evaluation_interval_seconds': self.research_config.get('evaluation_interval_seconds', 60),
            'last_run': self._research_last_run.isoformat() if self._research_last_run else None,
            'last_symbol': self._research_last_symbol,
            'last_entered': self._research_last_entered,
            'consecutive_errors': self._research_consecutive_errors,
            'last_error': self._research_last_error,
            'aggregate': aggregate,
            'status': status,
            'data_ready': data_ready,
        }

    async def _get_zombies(self) -> Dict[str, Any]:
        # Detect zombies using the farm's 7-14 DTE-aligned detection logic.
        if not self.paper_trader_farm:
            return {'zombies': [], 'total': 0, 'report': 'Farm not initialized'}
        await self._refresh_zombies_snapshot(force=False)
        return self._zombies_snapshot

    async def _get_followed_traders(self) -> List[Dict]:
        # Get the followed traders list from DB.
        return await self.db.get_followed_traders()

    async def _get_dashboard(self) -> Dict[str, Any]:
        # Get full system dashboard data for Telegram /dashboard command.
        now = datetime.now(timezone.utc)
        eastern = ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
        uptime_s = int((now - self._start_time).total_seconds())
        hours, remainder = divmod(uptime_s, 3600)
        minutes, _ = divmod(remainder, 60)

        # Task health
        def _age(ts):
            if not ts:
                return None
            return int((now - ts).total_seconds())

        research_age = _age(self._research_last_run)
        exit_age = _age(self._exit_monitor_last_run)
        health_age = _age(self._last_health_check)

        # Data freshness
        data_status = self._get_snapshot_section('data_status')
        if data_status is None:
            await self._refresh_status_snapshot(force=True)
            data_status = self._status_snapshot.get('data_status', {})

        # Farm stats
        farm = self.paper_trader_farm
        active_traders = len(farm.active_traders) if farm else 0
        open_positions = sum(
            len(t.open_positions) for t in farm.active_traders.values()
        ) if farm else 0
        total_configs = len(farm.trader_configs) if farm else 0

        # Market status
        is_weekday = now_et.weekday() < 5
        market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        market_open = is_weekday and market_open_time <= now_et <= market_close_time

        # Conditions
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()

        return {
            'uptime': f"{hours}h {minutes}m",
            'market_open': market_open,
            'market_time_et': now_et.strftime('%H:%M %Z'),
            'conditions_score': conditions.get('score', 'N/A'),
            'conditions_label': conditions.get('warmth_label', 'N/A'),
            'data_status': data_status,
            'research_loop_age_s': research_age,
            'research_errors': self._research_consecutive_errors,
            'research_last_error': self._research_last_error,
            'exit_monitor_age_s': exit_age,
            'health_check_age_s': health_age,
            'total_configs': total_configs,
            'active_traders': active_traders,
            'open_positions': open_positions,
            'today_opened': self._today_opened,
            'today_closed': self._today_closed,
            'today_expired': self._today_expired,
        }

    async def _get_data_status(self) -> Dict[str, Dict[str, Optional[str]]]:
        # Collect data freshness signals for key tables.
        tables = {
            "Detections": ("detections", 24 * 60 * 60),
            "Options IV": ("options_iv", 2 * 60 * 60),
            "Prices": ("price_snapshots", 10 * 60),
            "Health": ("system_health", 10 * 60),
        }
        latest = await self.db.get_latest_timestamps(
            [t[0] for t in tables.values()]
        )
        now = datetime.now(timezone.utc)
        eastern = ZoneInfo("America/New_York")
        age_since_start = int((now - self._start_time).total_seconds())
        status: Dict[str, Dict[str, Optional[str]]] = {}
        for label, (table, threshold) in tables.items():
            ts = latest.get(table)
            if not ts:
                if age_since_start < threshold:
                    status[label] = {
                        "status": "pending",
                        "last_seen_et": "N/A",
                        "age_human": None,
                    }
                    continue
                status[label] = {
                    "status": "missing",
                    "last_seen_et": "N/A",
                    "age_human": None,
                }
                continue
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                status[label] = {
                    "status": "missing",
                    "last_seen_et": "N/A",
                    "age_human": None,
                }
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_seconds = max(0, int((now - parsed).total_seconds()))
            status[label] = {
                "status": "ok" if age_seconds <= threshold else "stale",
                "last_seen_et": parsed.astimezone(eastern).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "age_human": self._format_age(age_seconds),
            }
        return status

    @staticmethod
    def _format_age(age_seconds: int) -> str:
        # Format age in human-friendly units.
        if age_seconds < 60:
            return f"{age_seconds}s ago"
        if age_seconds < 3600:
            return f"{age_seconds // 60}m ago"
        if age_seconds < 86400:
            return f"{age_seconds // 3600}h ago"
        return f"{age_seconds // 86400}d ago"
    
    async def _on_bybit_ticker(self, data: Dict) -> None:
        # Handle Bybit ticker update.
        data['exchange'] = 'bybit'
        await self._maybe_log_price_snapshot(
            exchange='bybit',
            symbol=data.get('symbol'),
            price=data.get('last_price'),
            volume=data.get('volume_24h'),
        )
        
        if 'volatility' in self.detectors:
            self._note_detector_activity(
                self.detectors['volatility'].name, kind="metric"
            )
            await self.detectors['volatility'].analyze(data)

    async def _on_coinbase_ticker(self, data: Dict) -> None:
        # Handle Coinbase ticker update.
        #
        # Coinbase is the PRIMARY truth source for crypto price data.
        # Updates truth-feed health tracking so OKX fallback knows to stay silent.
        data['exchange'] = 'coinbase'

        # Track truth-feed health per asset
        symbol = data.get('symbol', '')
        asset = 'BTC' if 'BTC' in symbol else 'ETH' if 'ETH' in symbol else None
        if asset and asset in self._truth_feed_health:
            health = self._truth_feed_health[asset]
            was_fallback = health["active_source"] != "coinbase"
            health["last_tick_mono"] = time.monotonic()
            health["active_source"] = "coinbase"
            if was_fallback:
                self.logger.info(
                    "Truth feed restored: %s -> Coinbase (primary)", asset
                )

        await self._maybe_log_price_snapshot(
            exchange='coinbase',
            symbol=data.get('symbol'),
            price=data.get('last_price'),
            volume=data.get('volume_24h'),
        )
        
        if 'volatility' in self.detectors:
            self._note_detector_activity(
                self.detectors['volatility'].name, kind="metric"
            )
            await self.detectors['volatility'].analyze(data)
    
    async def _on_yahoo_update(self, data: Dict) -> None:
        # Handle IBIT/BITO price update from Yahoo Finance.
        symbol = data.get('symbol')
        if not symbol:
            return
        data['source'] = 'yahoo'
        # Feed to any detector that tracks this symbol
        for name, detector in self.detectors.items():
            if name.startswith("etf_") and hasattr(detector, 'symbol'):
                if detector.symbol == symbol or detector.vol_proxy == symbol:
                    self._note_detector_activity(detector.name, kind="metric")
                    detection = await detector.analyze(data)
                    if detection:
                        await self._send_alert(detection)

    async def _maybe_log_price_snapshot(
        self,
        exchange: str,
        symbol: Optional[str],
        price: Optional[float],
        volume: Optional[float],
        min_interval_seconds: int = 60,
    ) -> None:
        # Record price snapshots at a controlled cadence.
        if not symbol or price is None:
            return
        now = datetime.now(timezone.utc)
        key = f"{exchange}:{symbol}"
        last_logged = self._last_price_snapshot.get(key)
        if last_logged and (now - last_logged).total_seconds() < min_interval_seconds:
            return
        await self.db.insert_price_snapshot(
            exchange=exchange,
            asset=symbol,
            price_type='spot',
            price=float(price),
            volume=volume,
        )
        self._last_price_snapshot[key] = now
    
    async def _on_funding_update(self, data: Dict) -> None:
        # Handle funding rate update from Bybit.
        self.logger.debug(f"Funding update: {data['symbol']} = {data['rate']:.4%}")
    
    async def _send_alert(self, detection: Dict) -> None:
        # Send alert for a detection.
        if not self.telegram:
            return
        if self.research_enabled and not self.research_alerts_enabled:
            return
        
        op_type = detection.get('opportunity_type')
        tier = detection.get('alert_tier', 2)
        
        self.logger.info(
            f"DETECTION: {op_type} - {detection.get('asset')} - "
            f"Edge: {detection.get('net_edge_bps', 0):.1f} bps (tier {tier})"
        )
        
        if op_type == 'options_iv':
            data = detection.get('detection_data', {})
            if data.get('is_data_only'):
                return
            await self.telegram.send_iv_alert(detection)
        elif op_type == 'etf_options':
            await self._send_etf_alert(detection)
    
    async def _send_paper_notification(self, message: str) -> None:
        # Send paper trade notification via Telegram.
        if self.telegram and (not self.research_enabled or self.research_alerts_enabled):
            try:
                await self.telegram.send_message(message, parse_mode="Markdown")
            except Exception as e:
                self.logger.warning(f"Failed to send paper notification: {e}")
    
    async def _send_etf_alert(self, detection: Dict) -> None:
        # Send ETF options opportunity alert.
        data = detection.get('detection_data', {})
        symbol = detection.get('asset', 'ETF')
        
        # Get conditions for the alert display
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        
        await self.telegram.send_alert(
            tier=1,  # High priority - actionable
            alert_type='options_iv',
            title=f"🎯 {symbol} OPTIONS OPPORTUNITY",
            details={
                f'{symbol} Price': f"${data.get('price', 0):.2f}",
                f'{symbol} 24h Change': f"{data.get('change_24h', 0):+.1f}%",
                'Proxy IV': f"{data.get('vol_iv', 0):.0f}%",
                'IV Rank': f"{data.get('iv_rank', 0):.0f}%",
                'Risk Flow': f"{conditions.get('risk_flow', 'N/A')}",
                'Market': '🟢 OPEN' if conditions.get('market_open') else '🔴 CLOSED',
                'Short Strike': f"${data.get('short_strike', 0):.0f}",
                'Long Strike': f"${data.get('long_strike', 0):.0f}",
            },
            action=f"Check broker for {symbol} put spread opportunity"
        )
    
    async def _poll_deribit(self) -> None:
        # Poll Deribit for options IV data.
        if not self.deribit_client:
            return
        
        interval = 60  # seconds
        
        while self._running:
            try:
                for currency in ['BTC', 'ETH']:
                    data = await self.deribit_client.get_atm_iv(currency)
                    if data:
                        # Feed to options IV detector
                        if 'options_iv' in self.detectors:
                            self._note_detector_activity(
                                self.detectors['options_iv'].name, kind="metric"
                            )
                            detection = await self.detectors['options_iv'].analyze(data)
                            if detection:
                                await self._send_alert(detection)
                        
                        # Feed BTC IV to all crypto detectors
                        if currency == 'BTC':
                            # Feed to detectors
                            for detector in self.detectors.values():
                                if hasattr(detector, 'vol_proxy') and detector.vol_proxy == 'BTC':
                                    self._note_detector_activity(detector.name, kind="metric")
                                    detector.update_proxy_iv(data.get('atm_iv', 0))
                            
                            # Feed BTC Greeks to AI Agent for research context
                            if self.ai_agent and data.get('greeks'):
                                self.ai_agent.context_injector.set_benchmark_greeks(data['greeks'])
                            
            except Exception as e:
                self.logger.error(
                    "Deribit polling error (%s): %s",
                    type(e).__name__, e,
                )
                self.logger.debug("Deribit polling error detail", exc_info=True)
                now = time.time()
                if (now - self._deribit_traceback_ts) >= 60:
                    self._deribit_traceback_ts = now
                    self.logger.error("Deribit polling traceback:\n%s", traceback.format_exc())
            
            await asyncio.sleep(interval)
    
    def _on_spread_signal(self, signal) -> None:
        # Handle spread signal from SpreadCandidateGenerator.
        #
        # Emits SignalEvent to signals.raw with strategy_id=PUT_SPREAD_V1.
        from .core.signals import SignalEvent as Phase3Signal, signal_to_dict
        
        # Convert to Phase 3 SignalEvent if needed
        if hasattr(signal, 'signal_id'):
            # Already a SignalEvent-like object
            self.event_bus.publish(TOPIC_SIGNALS_RAW, signal)
            self.logger.debug(
                "Spread signal emitted: %s %s credit=%.2f",
                signal.symbol if hasattr(signal, 'symbol') else "?",
                signal.direction if hasattr(signal, 'direction') else "?",
                signal.metadata.get('credit', 0) if hasattr(signal, 'metadata') else 0,
            )
    
    async def _poll_options_chains(self) -> None:
        # Poll Alpaca for options chain snapshots.
        #
        # Publishes OptionChainSnapshotEvent to options.chains topic.
        # SpreadCandidateGenerator is already subscribed via event bus.
        # Runs in its own task; backoff and interval are independent of
        # _poll_tastytrade_options_chains (no shared counter). Resumes
        # normal interval after transient failure once consecutive_errors
        # is reset.
        if not self.alpaca_options:
            return

        interval = getattr(self, '_alpaca_options_poll_interval', 60)
        symbols = getattr(self, '_alpaca_options_symbols', _DEFAULT_OPTIONS_SYMBOLS)
        min_dte = getattr(self, '_alpaca_options_min_dte', 7)
        max_dte = getattr(self, '_alpaca_options_max_dte', 21)

        # Per-loop state (not shared with Tastytrade polling)
        last_warning_ts = 0
        warning_count = 0
        consecutive_errors = 0

        while self._running:
            try:
                # Check market hours — skip off-hours if configured
                if self._mh_cfg.get('off_hours_disable_options_snapshots', False):
                    if not self._is_us_market_open():
                        await asyncio.sleep(interval)
                        continue

                for symbol in symbols:
                    # Get expirations in DTE range
                    expirations = await self.alpaca_options.get_expirations_in_range(
                        symbol, min_dte=min_dte, max_dte=max_dte
                    )

                    for exp_date, dte in expirations:
                        snapshot = await self.alpaca_options.build_chain_snapshot(
                            symbol, exp_date
                        )
                        if snapshot:
                            # Publish to event bus (TapeRecorder + SpreadGenerator subscribe)
                            self.event_bus.publish(TOPIC_OPTIONS_CHAINS, snapshot)
                            self.logger.debug(
                                "Alpaca chain: %s exp=%s DTE=%d puts=%d calls=%d provider=%s",
                                symbol, exp_date, dte,
                                len(snapshot.puts), len(snapshot.calls),
                                snapshot.provider,
                            )
                        else:
                            warning_count += 1
                            now = time.time()
                            if now - last_warning_ts > 60:
                                self.logger.warning(
                                    "Alpaca empty chain for %s exp=%s (suppressed %d similar)",
                                    symbol, exp_date, warning_count,
                                )
                                last_warning_ts = now
                                warning_count = 0

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(
                    "Alpaca options polling error (consecutive=%d) (%s): %s",
                    consecutive_errors, type(e).__name__, e,
                )
                self.logger.debug("Alpaca options polling error detail", exc_info=True)
                if consecutive_errors >= 3:
                    backoff = min(interval * consecutive_errors, 300)
                    self.logger.warning(
                        "Alpaca options backing off for %ds after %d consecutive errors",
                        backoff, consecutive_errors,
                    )
                    await asyncio.sleep(backoff)
                    continue

            await asyncio.sleep(interval)

    async def _start_dxlink_greeks_streamer(self) -> None:
        # Obtain a DXLink token and run a Greeks streamer in the background.
        #
        # Uses the already-authenticated TastytradeOptionsConnector's REST
        # client to fetch a DXLink streaming token, then subscribes to
        # Greeks events for the configured option symbols.  Events are
        # forwarded to ``_on_dxlink_greeks_event`` which populates
        # ``_greeks_cache``.
        if not self.tastytrade_options:
            return

        # Use option-level symbols for Greeks (underlyings alone do not receive Greeks from DXLink)
        symbols = getattr(self, "_dxlink_greeks_symbols", None) or getattr(self, "_tastytrade_options_symbols", [])
        if not symbols:
            return

        try:
            # Get authenticated REST client from the options connector
            client = self.tastytrade_options._ensure_client()
            quote_info = client.get_api_quote_token()
            dxlink_url = quote_info["dxlink-url"]
            dxlink_token = quote_info["token"]

            # Build DXLink option symbols from the first snapshot fetch
            # For now subscribe to underlying symbols for Quote events
            # plus Greeks event type so the streamer asks for Greeks
            self._dxlink_streamer = TastytradeStreamer(
                dxlink_url=dxlink_url,
                token=dxlink_token,
                symbols=symbols,
                on_event=self._on_dxlink_greeks_event,
                event_types=["Greeks"],
            )
            self.logger.info(
                "DXLink Greeks streamer starting for %s", symbols,
            )
            await self._dxlink_streamer.run_forever()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error(
                "DXLink Greeks streamer failed (%s): %s",
                type(exc).__name__, exc,
            )
            self.logger.debug("DXLink Greeks streamer failure detail", exc_info=True)

    def _on_dxlink_greeks_event(self, event: Any) -> None:
        # Handle DXLink Greeks events by caching IV and Greeks.
        #
        # Called from the DXLink streamer callback. Populates the
        # in-memory GreeksCache so that subsequent snapshot polls
        # can enrich ``atm_iv`` from cached provider IV.
        #
        # No DB writes — memory only.
        event_symbol = getattr(event, "event_symbol", None)
        volatility = getattr(event, "volatility", None)
        if not event_symbol or volatility is None:
            return

        recv_ts_ms = getattr(event, "receipt_time", None)
        self._greeks_cache.update(
            event_symbol=event_symbol,
            volatility=volatility,
            recv_ts_ms=recv_ts_ms,
            delta=getattr(event, "delta", None),
            gamma=getattr(event, "gamma", None),
            theta=getattr(event, "theta", None),
            vega=getattr(event, "vega", None),
        )
        self._iv_consensus.observe_dxlink_greeks(event, recv_ts_ms=recv_ts_ms)

    def get_consensus_atm_iv(self, underlying: str, option_type: str, expiration_ms: int, as_of_ms: int):
        # Return consensus ATM IV result for strategy/risk modules.
        return self._iv_consensus.get_atm_consensus(
            underlying=underlying,
            option_type=option_type,
            expiration_ms=expiration_ms,
            as_of_ms=as_of_ms,
        )

    def get_consensus_contract_iv(self, underlying: str, expiration_ms: int, option_type: str, strike: float, as_of_ms: int):
        # Return consensus per-contract IV/greeks for strategy/risk modules.
        from .core.iv_consensus import ContractKey

        return self._iv_consensus.get_contract_consensus(
            ContractKey(
                underlying=underlying,
                expiration_ms=expiration_ms,
                option_type=option_type.upper(),
                strike=float(strike),
            ),
            as_of_ms=as_of_ms,
        )

    async def _poll_tastytrade_options_chains(self) -> None:
        # Poll Tastytrade for options chain snapshots via REST.
        #
        # Publishes OptionChainSnapshotEvents to options.chains topic.
        # Own task; backoff and interval independent of Alpaca polling.
        # If one provider fails the other continues; normal interval
        # resumes after consecutive_errors is reset.
        if not self.tastytrade_options:
            return

        interval = getattr(self, '_tastytrade_options_poll_interval', 60)
        symbols = getattr(self, '_tastytrade_options_symbols', [])
        min_dte = getattr(self, '_tastytrade_options_min_dte', 7)
        max_dte = getattr(self, '_tastytrade_options_max_dte', 21)

        # Per-loop state (not shared with Alpaca polling)
        last_warning_ts = 0
        warning_count = 0
        consecutive_errors = 0

        while self._running:
            try:
                # Check market hours — skip off-hours if configured
                if self._mh_cfg.get('off_hours_disable_options_snapshots', False):
                    if not self._is_us_market_open():
                        await asyncio.sleep(interval)
                        continue

                for symbol in symbols:
                    try:
                        # Underlying price for ATM IV enrichment: Alpaca Options quote, else latest bar close from DB (alpaca bars)
                        underlying_price = 0.0
                        if self.alpaca_options:
                            try:
                                price, _, _ = await self.alpaca_options.get_underlying_quote(symbol)
                                underlying_price = price
                            except Exception:
                                pass

                        if underlying_price <= 0 and self.db:
                            try:
                                close = await self.db.get_latest_bar_close("alpaca", symbol, bar_duration=60)
                                if close and close > 0:
                                    underlying_price = close
                            except Exception:
                                pass

                        # Run sync REST call in thread to avoid blocking event loop
                        snapshots = await asyncio.to_thread(
                            self.tastytrade_options.build_snapshots_for_symbol,
                            symbol,
                            min_dte=min_dte,
                            max_dte=max_dte,
                            underlying_price=underlying_price,
                        )
                        for snapshot in snapshots:
                            self._iv_consensus.observe_public_snapshot(snapshot, recv_ts_ms=snapshot.recv_ts_ms)
                            # Enrich snapshot with ATM IV from consensus engine
                            snapshot = enrich_snapshot_iv(snapshot, self._iv_consensus)
                            self.event_bus.publish(TOPIC_OPTIONS_CHAINS, snapshot)
                            self.logger.debug(
                                "Tastytrade chain: %s exp_ms=%d puts=%d calls=%d provider=%s atm_iv=%s",
                                snapshot.symbol, snapshot.expiration_ms,
                                len(snapshot.puts), len(snapshot.calls),
                                snapshot.provider, snapshot.atm_iv,
                            )

                        if not snapshots:
                            warning_count += 1
                            now = time.time()
                            if now - last_warning_ts > 120:
                                self.logger.warning(
                                    "Tastytrade empty snapshots for %s (suppressed %d similar)",
                                    symbol, warning_count,
                                )
                                last_warning_ts = now
                                warning_count = 0

                    except Exception as e:
                        self.logger.error(
                            "Tastytrade polling error for %s (%s): %s",
                            symbol, type(e).__name__, e,
                        )
                        self.logger.debug("Tastytrade polling error detail for %s", symbol, exc_info=True)

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(
                    "Tastytrade options polling loop error (consecutive=%d) (%s): %s",
                    consecutive_errors, type(e).__name__, e,
                )
                self.logger.debug("Tastytrade options loop error detail", exc_info=True)
                # Back off if we're getting repeated failures
                if consecutive_errors >= 3:
                    backoff = min(interval * consecutive_errors, 300)
                    self.logger.warning(
                        "Tastytrade backing off for %ds after %d consecutive errors",
                        backoff, consecutive_errors,
                    )
                    await asyncio.sleep(backoff)
                    continue

            await asyncio.sleep(interval)

    async def _poll_public_options_chains(self) -> None:
        # Poll Public.com Greeks and publish options chain snapshots.
        if not self.public_options:
            return

        interval = getattr(self, '_public_options_poll_interval', 60)
        symbols = getattr(self, '_public_options_symbols', _DEFAULT_OPTIONS_SYMBOLS)
        min_dte = getattr(self, '_public_options_min_dte', 7)
        max_dte = getattr(self, '_public_options_max_dte', 21)
        consecutive_errors = 0

        while self._running:
            try:
                if self._mh_cfg.get('off_hours_disable_options_snapshots', False):
                    if not self._is_us_market_open():
                        await asyncio.sleep(interval)
                        continue

                for symbol in symbols:
                    try:
                        underlying_price = 0.0
                        if self.alpaca_options:
                            try:
                                p, _, _ = await self.alpaca_options.get_underlying_quote(symbol)
                                underlying_price = p
                            except Exception:
                                pass

                        if underlying_price <= 0 and self.db:
                            try:
                                close = await self.db.get_latest_bar_close("alpaca", symbol, bar_duration=60)
                                if close and close > 0:
                                    underlying_price = close
                            except Exception:
                                pass

                        snapshots = await self.public_options.build_snapshots_for_symbol(
                            symbol,
                            min_dte=min_dte,
                            max_dte=max_dte,
                            underlying_price=underlying_price,
                        )
                        for snapshot in snapshots:
                            self._iv_consensus.observe_public_snapshot(snapshot, recv_ts_ms=snapshot.recv_ts_ms)
                            snapshot = enrich_snapshot_iv(snapshot, self._iv_consensus)
                            self.event_bus.publish(TOPIC_OPTIONS_CHAINS, snapshot)
                            self.logger.debug(
                                "Public chain: %s exp_ms=%d puts=%d calls=%d atm_iv=%s",
                                snapshot.symbol,
                                snapshot.expiration_ms,
                                len(snapshot.puts),
                                len(snapshot.calls),
                                snapshot.atm_iv,
                            )
                    except Exception as exc:
                        self.logger.error(
                            "Public options polling error for %s (%s): %s",
                            symbol, type(exc).__name__, exc,
                        )
                        self.logger.debug("Public options polling error detail for %s", symbol, exc_info=True)

                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                self.logger.error(
                    "Public options polling loop error (consecutive=%d) (%s): %s",
                    consecutive_errors, type(exc).__name__, exc,
                )
                self.logger.debug("Public options loop error detail", exc_info=True)
                if consecutive_errors >= 3:
                    backoff = min(interval * consecutive_errors, 300)
                    await asyncio.sleep(backoff)
                    continue

            await asyncio.sleep(interval)

    async def _get_current_spread_prices(self) -> Dict[str, Dict[str, float]]:
        # Get current spread prices for open positions to evaluate exits.
        prices: Dict[str, Dict[str, float]] = {}
        if not self.paper_trader_farm:
            return prices

        for trader_id, trader in self.paper_trader_farm.active_traders.items():
            for trade in trader.open_positions:
                symbol = trade.symbol
                if symbol not in prices:
                    prices[symbol] = {}

                current_price = None
                detector = self.detectors.get(symbol.lower())
                if detector and hasattr(detector, '_current_ibit_data') and detector._current_ibit_data:
                    current_price = detector._current_ibit_data.get('price')
                if current_price is None:
                    continue

                try:
                    if '/' not in trade.strikes:
                        continue
                    parts = trade.strikes.replace('$', '').split('/')
                    short_strike = float(parts[0])
                    long_strike = float(parts[1])
                    spread_width = short_strike - long_strike
                    entry_credit = trade.entry_credit

                    # Guard: skip positions with invalid spread width or credit
                    if spread_width <= 0 or not entry_credit or entry_credit <= 0:
                        continue

                    if current_price >= short_strike:
                        otm_pct = (current_price - short_strike) / current_price if current_price > 0 else 0
                        decay_factor = max(0.0, 1.0 - otm_pct * 10)
                        current_value = entry_credit * decay_factor * 0.3
                    elif current_price <= long_strike:
                        current_value = spread_width
                    else:
                        itm_pct = (short_strike - current_price) / spread_width
                        current_value = entry_credit + (spread_width - entry_credit) * itm_pct

                    prices[symbol][trade.id] = max(0.0, current_value)
                except (ValueError, IndexError, ZeroDivisionError):
                    continue

        return prices

    async def _run_market_close_snapshot(self) -> None:
        # Take gap risk snapshots at market close (4 PM ET) on weekdays.
        if not self.gap_risk_tracker:
            return
        eastern = ZoneInfo("America/New_York")
        last_snapshot_date = None

        while self._running:
            try:
                now_et = datetime.now(eastern)
                today = now_et.date()
                is_weekday = now_et.weekday() < 5
                past_close = now_et.hour >= 16

                if is_weekday and past_close and last_snapshot_date != today:
                    btc_price_data = await self._get_btc_price()
                    btc_price = btc_price_data.get('price', 0) if btc_price_data else 0

                    ibit_price = None
                    bito_price = None
                    ibit_det = self.detectors.get('ibit')
                    if ibit_det and hasattr(ibit_det, '_current_ibit_data') and ibit_det._current_ibit_data:
                        ibit_price = ibit_det._current_ibit_data.get('price')
                    bito_det = self.detectors.get('bito')
                    if bito_det and hasattr(bito_det, '_current_ibit_data') and bito_det._current_ibit_data:
                        bito_price = bito_det._current_ibit_data.get('price')

                    if btc_price > 0:
                        await self.gap_risk_tracker.snapshot_market_close(
                            btc_price=btc_price,
                            ibit_price=ibit_price,
                            bito_price=bito_price,
                        )
                        last_snapshot_date = today
                        self.logger.info(f"Market close snapshot taken: BTC=${btc_price:,.0f}")
            except Exception as e:
                self.logger.error(f"Market close snapshot error: {e}")
            await asyncio.sleep(300)

    async def _cleanup_zombie_positions(self) -> None:
        # Close orphaned positions from previous runs that are still 'open' in DB.
        #
        # Uses close_timestamp (not closed_at) and close_reason columns which
        # are created by the migration in paper_trader_farm._create_tables.
        try:
            row = await self.db.fetch_one(
                "SELECT COUNT(*) as cnt FROM paper_trades WHERE status = 'open'"
            )
            zombie_count = row['cnt'] if row else 0
            if zombie_count == 0:
                return

            self.logger.info(f"Found {zombie_count:,} zombie positions from previous runs, marking as expired")
            now_ts = datetime.now(timezone.utc).isoformat()

            # Use close_timestamp (the actual column name) and close_reason (added by migration)
            try:
                await self.db.execute(
                    """UPDATE paper_trades SET status = 'expired',
                       close_reason = 'system_restart_cleanup',
                       close_timestamp = ?
                       WHERE status = 'open'""",
                    (now_ts,)
                )
            except Exception:
                # Fallback if close_reason column doesn't exist yet
                await self.db.execute(
                    """UPDATE paper_trades SET status = 'expired',
                       close_timestamp = ?
                       WHERE status = 'open'""",
                    (now_ts,)
                )
            self.logger.info(f"Cleaned up {zombie_count:,} zombie positions")
        except Exception as e:
            self.logger.error(f"Failed to cleanup zombie positions: {e}")

    async def _run_exit_monitor(self) -> None:
        # Independent task: check exits and expirations every 30 seconds.
        #
        # Decoupled from the research signal loop so exits still happen even
        # if signal evaluation crashes.
        #
        # On exception, reports the error to the farm's runaway safety
        # tracker so repeated failures can halt new entries.
        if not self.paper_trader_farm:
            return
        interval = 30

        while self._running:
            try:
                # Check exits based on current prices
                current_prices = await self._get_current_spread_prices()
                if current_prices:
                    closed_trades = await self.paper_trader_farm.check_exits(current_prices)
                    if closed_trades:
                        n = len(closed_trades)
                        self._today_closed += n
                        self.logger.info(f"Exit monitor: {n} trades closed")
                        # Add to summary queue instead of sending individual summary
                        self._recent_farm_exits.extend(closed_trades)

                # Check expirations
                eastern = ZoneInfo("America/New_York")
                today_et = datetime.now(eastern).strftime('%Y-%m-%d')
                expired_trades = await self.paper_trader_farm.expire_positions(today_et)
                if expired_trades:
                    n = len(expired_trades)
                    self._today_expired += n
                    self.logger.info(f"Exit monitor: {n} trades expired")

                self._exit_monitor_last_run = datetime.now(timezone.utc)
            except Exception as e:
                self.logger.error(f"Exit monitor error: {e}")
                # Feed error into farm runaway safety tracker
                if self.paper_trader_farm:
                    self.paper_trader_farm.record_exit_error()
            await asyncio.sleep(interval)

    async def _run_research_farm(self) -> None:
        # Continuously evaluate farm signals for research.
        #
        # Exit checking is handled by the separate _run_exit_monitor task,
        # so this loop only handles signal evaluation and new entries.
        if not self.paper_trader_farm:
            return
        interval = int(self.research_config.get('evaluation_interval_seconds', 60))
        interval = max(10, interval)

        while self._running and self.research_enabled:
            # Always update the timestamp so we can see the loop is alive
            self._research_last_run = datetime.now(timezone.utc)
            try:
                # Gather market conditions
                conditions = {}
                if self.conditions_monitor:
                    conditions = await self.conditions_monitor.get_current_conditions()
                conditions_score = int(conditions.get('score', 5))
                conditions_label = conditions.get('warmth_label', 'neutral')
                btc_change = float(conditions.get('btc_change', 0))
                btc_change_5d = float(conditions.get('btc_change_5d', 0))
                timestamp = datetime.now(timezone.utc).isoformat()

                total_entered = 0
                for key in ('ibit', 'bito'):
                    detector = self.detectors.get(key)
                    if not detector:
                        continue
                    try:
                        signal = await asyncio.to_thread(
                            detector.get_research_signal,
                            conditions_score=conditions_score,
                            conditions_label=conditions_label,
                            btc_change_24h_pct=btc_change,
                            btc_change_5d_pct=btc_change_5d,
                            timestamp=timestamp,
                        )
                    except Exception as e:
                        self.logger.error(f"Signal generation failed for {key}: {e}")
                        continue
                    if not signal:
                        continue
                    trades = await self.paper_trader_farm.evaluate_signal(
                        symbol=signal['symbol'],
                        signal_data=signal,
                    )
                    entered = len(trades)
                    total_entered += entered
                    self._research_last_symbol = signal['symbol']

                    # Alert if any followed traders entered
                    if trades:
                        self._recent_farm_entries.extend(trades)
                        if self.telegram:
                            await self._alert_followed_trades(trades, signal)

                self._research_last_entered = total_entered
                self._today_opened += total_entered
                self._research_consecutive_errors = 0
                self._research_last_error = None
                await self._maybe_promote_configs()
                await self._run_uniformity_check()
            except Exception as e:
                self._research_consecutive_errors += 1
                self._research_last_error = str(e)
                self.logger.error(
                    f"Research farm error (#{self._research_consecutive_errors}): {e}"
                )
                # Alert via telegram if errors persist
                if self._research_consecutive_errors == 5 and self.telegram:
                    try:
                        await self.telegram.send_message(
                            f"⚠️ <b>Research Loop Degraded</b>\n"
                            f"5 consecutive errors.\n"
                            f"Last error: <code>{str(e)[:200]}</code>"
                        )
                    except Exception:
                        pass

            await asyncio.sleep(interval)

    async def _maybe_promote_configs(self) -> None:
        # Promote top-performing configs after research window.
        if self._research_promoted:
            return
        if not self.research_config.get('auto_promote_enabled', False):
            return
        promote_after_days = int(self.research_config.get('promote_after_days', 60))
        days_since_start = (datetime.now(timezone.utc) - self._start_time).days
        if days_since_start < promote_after_days:
            return

        window_days = int(self.research_config.get('promotion_window_days', promote_after_days))
        min_trades = int(self.research_config.get('promotion_min_trades', 30))
        min_total_pnl = float(self.research_config.get('promotion_min_total_pnl', 250.0))
        min_avg_pnl = float(self.research_config.get('promotion_min_avg_pnl', 5.0))
        min_win_rate = float(self.research_config.get('promotion_min_win_rate', 55.0))
        top_n = int(self.research_config.get('promotion_top_n', 5))

        performance = await self.db.get_trader_performance(days=window_days)
        eligible = [
            p for p in performance
            if p.get('total_trades', 0) >= min_trades
            and p.get('total_pnl', 0) >= min_total_pnl
            and p.get('avg_pnl', 0) >= min_avg_pnl
            and p.get('win_rate', 0) >= min_win_rate
        ]
        if not eligible:
            return
        eligible.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
        promoted_ids = [p['trader_id'] for p in eligible[:top_n]]

        if self.paper_trader_farm:
            self.paper_trader_farm.set_promoted_traders(promoted_ids)
        self._research_promoted = True

        if self.research_config.get('live_mode_after_promotion', False):
            self.research_enabled = False
            for key in ('ibit', 'bito'):
                detector = self.detectors.get(key)
                if detector:
                    detector.paper_trading_enabled = True

    async def _run_periodic_farm_summary(self) -> None:
        # Background task: Periodically send consolidated farm activity summaries.
        while self._running:
            try:
                # Summary interval: 30 minutes
                interval = 30 * 60
                await asyncio.sleep(interval)
                
                if not self.telegram or not self.paper_trader_farm:
                    continue
                    
                # Collect activity
                entries = self._recent_farm_entries.copy()
                exits = self._recent_farm_exits.copy()
                self._recent_farm_entries.clear()
                self._recent_farm_exits.clear()
                
                if not entries and not exits:
                    continue
                    
                lines = ["🚜 <b>LIVE FARM ACTIVITY</b>"]
                
                if entries:
                    symbols = set(t.symbol for t in entries)
                    lines.append(f"• <b>New Entries</b>: {len(entries)} (Symbols: {', '.join(symbols)})")
                
                if exits:
                    total_pnl = sum(t.realized_pnl for t in exits)
                    wins = sum(1 for t in exits if t.realized_pnl > 0)
                    lines.append(f"• <b>Realized P&L</b>: ${total_pnl:+.2f} ({wins}/{len(exits)} wins)")
                
                lines.append(f"\n<i>Summary of the last 30 minutes.</i>")
                
                await self.telegram.send_tiered_message("\n".join(lines), priority=2, key="periodic_farm_summary")
                self._farm_summary_last_run = time.time()
                
            except Exception as e:
                self.logger.error(f"Farm summary task error: {e}")
                await asyncio.sleep(60)

    async def _alert_followed_trades(self, trades: list, signal: dict) -> None:
        # Send Telegram alert when followed traders enter positions.
        if not self.telegram:
            return
        try:
            followed = await self.db.get_followed_traders()
            if not followed:
                return
            followed_ids = {t['trader_id'] for t in followed}
            matched = [t for t in trades if hasattr(t, 'trader_id') and t.trader_id in followed_ids]
            if not matched:
                return

            for trade in matched[:3]:  # Limit to 3 to avoid spam
                msg = (
                    f"🌟 <b>FOLLOWED TRADER ENTRY</b>\n"
                    f"Trader: <code>{trade.trader_id[:8]}</code>\n"
                    f"Strategy: {trade.strategy_type}\n"
                    f"Symbol: {trade.symbol} at {trade.entry_price:.2f}\n"
                    f"Credit: ${trade.entry_credit:.2f} | PoP: {trade.entry_pop:.1f}%"
                )
                await self.telegram.send_tiered_message(msg, priority=2, key=f"follow_{trade.trader_id}")
        except Exception as e:
            self.logger.warning(f"Failed to alert followed trades: {e}")

    async def _send_exit_summary(self, trades: list) -> None:
        # Send consolidated summary of closed trades.
        if not self.telegram:
            return
        
        # Group by symbol/strategy to avoid spam
        # We only really care about the outcome
        total_pnl = sum(t.realized_pnl for t in trades)
        win_count = sum(1 for t in trades if t.realized_pnl > 0)
        
        # If mass-closing (e.g. expiration), summarising is better
        if len(trades) > 3:
            msg = (
                f"💰 <b>TRADES CLOSED</b> ({len(trades)})\n"
                f"Realized P&L: <b>${total_pnl:+.2f}</b>\n"
                f"Win Rate: {win_count}/{len(trades)} ({win_count/len(trades):.0%})"
            )
            await self.telegram.send_message(msg)
            return

        # For individual/few trades, show details
        for t in trades:
            emoji = "🟢" if t.realized_pnl > 0 else "🔴"
            credit_text = f"${t.entry_credit:.2f}"
            
            # Calculate return % on risk/collateral
            # Simplify: return on credit for now or just absolute PnL
            
            msg = (
                f"{emoji} <b>TRADE CLOSED: {t.symbol}</b>\n"
                f"{t.strategy_type} {t.strikes}\n"
                f"Credit: {credit_text} → Closed: ${t.close_price:.2f}\n"
                f"<b>P&L: ${t.realized_pnl:+.2f}</b>"
            )
            await self.telegram.send_message(msg)

    async def _handle_soak_alert(self, severity: str, guard: str, message: str) -> None:
        # Handle soak guard alerts with filtering.
        # Log everything to disk
        self.logger.warning(f"SOAK GUARD [{severity}] {guard}: {message}")
        
        # Filter for Telegram
        # User requested SILENT soak guards unless critical
        if severity != "ALERT":
            return
            
        # Only specific critical guards go to Telegram
        critical_guards = {
            'ingestion_paused', 
            'disk_fatigue', 
            'params_corruption',
            'bar_liveness'
        }
        
        if guard in critical_guards or 'corruption' in message.lower():
            if self.telegram:
                await self.telegram.send_message(
                    f"🛡️ <b>SYSTEM CRITICAL</b>\n"
                    f"Guard: {guard}\n"
                    f"Message: {message}"
                )

    async def _run_uniformity_check(self) -> None:
        # Run uniformity check every N evaluations to detect convergence bugs.
        if not hasattr(self, '_uniformity_check_count'):
            self._uniformity_check_count = 0
        self._uniformity_check_count += 1

        # Run every 10 evaluations to avoid overhead
        if self._uniformity_check_count % 10 != 0:
            return

        if not self.paper_trader_farm:
            return

        # Gather recent trades from active traders
        recent_trades = []
        for trader_id, trader in self.paper_trader_farm.active_traders.items():
            for pos in trader.open_positions:
                recent_trades.append({
                    'trader_id': trader_id,
                    'strategy_type': pos.strategy if hasattr(pos, 'strategy') else '',
                    'strikes': pos.strikes if hasattr(pos, 'strikes') else '',
                    'expiry': pos.expiry if hasattr(pos, 'expiry') else '',
                    'entry_credit': pos.entry_credit if hasattr(pos, 'entry_credit') else 0,
                    'contracts': pos.contracts if hasattr(pos, 'contracts') else 0,
                })

        if len(recent_trades) < 20:
            return

        try:
            results = await run_uniformity_check(
                trades=recent_trades,
                db=self.db,
            )
            alerts = [r for r in results if r.get('is_alert')]
            if alerts and self.telegram:
                from .analysis.uniformity_monitor import format_uniformity_report
                report = format_uniformity_report(results)
                msg = f"⚠️ <b>Uniformity Alert</b>\n<pre>{report[:1500]}</pre>"
                await self.telegram.send_message(msg)
        except Exception as e:
            self.logger.warning(f"Uniformity check error: {e}")

    async def _log_iv_status(self) -> None:
        # Log whether we are receiving IV (Consensus or DXLink) every 60s.
        interval = 60
        stale_seconds = 300  # consider "stale" if no update in 5 min
        while self._running:
            await asyncio.sleep(interval)
            
            try:
                # 1. Check Consensus Engine (Merged/Preferred)
                n_con = self._iv_consensus.size if self._iv_consensus else 0
                last_con_ms = self._iv_consensus.last_update_ms if self._iv_consensus else 0
                
                # 2. Check Legacy DXLink Cache
                n_dx = self._greeks_cache.size if self._greeks_cache else 0
                last_dx_ms = self._greeks_cache.last_update_ms if self._greeks_cache else 0
                
                now_ms = int(time.time() * 1000)
                
                # Determine combined status
                last_update_ms = max(last_con_ms, last_dx_ms)
                age_s = (now_ms - last_update_ms) / 1000.0 if last_update_ms else float("inf")
                
                if n_con == 0 and n_dx == 0:
                    self.logger.info("IV: not receiving (no data in consensus or cache)")
                elif age_s > stale_seconds:
                    sources = []
                    if n_con > 0: sources.append(f"Consensus:{n_con}")
                    if n_dx > 0: sources.append(f"DXLink:{n_dx}")
                    self.logger.info(
                        "IV: stale (%s, last update %.0fs ago)",
                        " + ".join(sources), age_s,
                    )
                else:
                    sources = []
                    if n_con > 0: sources.append(f"Consensus:{n_con}")
                    if n_dx > 0: sources.append(f"DXLink:{n_dx}")
                    self.logger.info(
                        "IV: receiving (%s, last %.0fs ago)",
                        " + ".join(sources), age_s,
                    )
            except Exception as e:
                self.logger.debug("IV status check failed: %s", e)

    async def _health_check(self) -> None:
        # Periodic health check (5 min) and 60-second heartbeat summary.
        health_interval = 300
        heartbeat_interval = int(self.config.get('monitoring', {}).get('heartbeat_interval', 60))
        last_heartbeat = 0.0
        last_health = 0.0

        while self._running:
            now = time.time()

            # 60-second heartbeat line
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                try:
                    farm = self.paper_trader_farm
                    active_traders = len(farm.active_traders) if farm else 0
                    open_positions = sum(
                        len(t.open_positions) for t in farm.active_traders.values()
                    ) if farm else 0
                    bybit_health = self.bybit_ws.get_health_status() if self.bybit_ws else {}
                    bybit_connected = bybit_health.get('extras', {}).get('connected', False)
                    bybit_str = "connected" if bybit_connected else "disconnected"
                    msg_age = bybit_health.get('seconds_since_last_message')
                    msg_age_str = f"{msg_age:.0f}s" if msg_age is not None else "N/A"
                    db_size = os.path.getsize(str(self.db.db_path)) / (1024 * 1024) if self.db.db_path.exists() else 0

                    risk_val = "N/A"
                    if self.global_risk_flow_updater and self.global_risk_flow_updater._last_value is not None:
                        risk_val = f"{self.global_risk_flow_updater._last_value:.4f}"

                    self.logger.info(
                        f"[HEARTBEAT] uptime={uptime_seconds():.0f}s "
                        f"bybit={bybit_str} last_msg={msg_age_str} "
                        f"traders={active_traders} positions={open_positions} "
                        f"risk_flow={risk_val} db={db_size:.0f}MB"
                    )
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")

            # 5-minute health check (DB write)
            if now - last_health >= health_interval:
                last_health = now
                try:
                    bybit_connected = self.bybit_ws.is_connected if self.bybit_ws else False
                    self._last_health_check = datetime.now(timezone.utc)
                    await self.db.insert_health_check(
                        component="bybit_ws",
                        status="connected" if bybit_connected else "disconnected",
                    )
                    await self.db.insert_health_check(
                        component="detectors",
                        status=f"active_{len(self.detectors)}",
                    )
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")

            try:
                await self._refresh_status_snapshot()
            except Exception as e:
                self.logger.debug(f"Status snapshot refresh error: {e}")

            try:
                await self._refresh_zombies_snapshot()
            except Exception as e:
                self.logger.debug(f"Zombies snapshot refresh error: {e}")

            await asyncio.sleep(10)  # Check every 10s for heartbeat granularity

    async def _run_market_session_monitor(self) -> None:
        # Monitor market open/close transitions and send notifications.
        eastern = ZoneInfo("America/New_York")

        while self._running:
            try:
                now_et = datetime.now(eastern)
                today = now_et.date()
                is_weekday = now_et.weekday() < 5
                market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                is_open = is_weekday and market_open_time <= now_et <= market_close_time

                # Detect market OPEN transition
                if is_open and not self._market_was_open and self._last_market_open_date != today:
                    self._last_market_open_date = today
                    self._today_opened = 0
                    self._today_closed = 0
                    self._today_expired = 0
                    await self._send_market_open_notification(now_et)

                # Detect market CLOSE transition
                if not is_open and self._market_was_open and self._last_market_close_date != today:
                    self._last_market_close_date = today
                    await self._send_market_close_notification(now_et)

                self._market_was_open = is_open
            except Exception as e:
                self.logger.error(f"Market session monitor error: {e}")
            await asyncio.sleep(30)

    async def _send_market_open_notification(self, now_et: datetime) -> None:
        # Send notification when market opens.
        if not self.telegram:
            return
        
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        
        score = conditions.get('score', 5)
        label = str(conditions.get('warmth_label', 'neutral')).upper()
        btc_iv = conditions.get('btc_iv', 'N/A')
        iv_rank = conditions.get('iv_rank', 'N/A')
        
        # Risk Flow status (Global Risk Flow)
        risk_flow = conditions.get('risk_flow', 'neutral')
        risk_icon = "🔵" if risk_flow == 'neutral' else ("🟢" if risk_flow == 'risk-on' else "🔴")

        # Current regime (vol/trend/risk from regime detector; bar-driven, so "last known" at open)
        regime_line = self._format_current_regime_for_telegram()

        header = f"🔔 <b>MARKET BRIEFING</b> — {now_et.strftime('%b %d, %Y')}"
        lines = [
            header,
            "",
            f"🌡️ <b>Conditions: {score}/10 {label}</b>",
            f"📈 BTC IV: {btc_iv}% (Rank: {iv_rank}%)",
            f"{risk_icon} Global Risk Flow: {str(risk_flow).upper()}",
            regime_line,
            "",
        ]

        if self.news_sentiment_updater:
            news_payload = self.news_sentiment_updater.get_last_payload()
            try:
                news_payload = await self.news_sentiment_updater.update()
            except Exception as exc:
                self.logger.warning("Failed to refresh news sentiment for market open: %s", exc)
            lines.append(format_news_sentiment_telegram(news_payload))
            lines.append("")

        # SPY, IBIT, QQQ prices (user-facing tickers)
        price_lines = await self._format_briefing_prices()
        if price_lines:
            lines += price_lines + [""]

        await self.telegram.send_tiered_message("\n".join(lines), priority=2, key="market_open")

    async def _get_sentiment_summary(self) -> str:
        # Build a combined sentiment summary for Telegram /sentiment command.
        lines = ["<b>📊 Sentiment Snapshot</b>", ""]

        # News sentiment
        news_payload = None
        if self.news_sentiment_updater:
            news_payload = self.news_sentiment_updater.get_last_payload()
            if not news_payload:
                try:
                    news_payload = await self.news_sentiment_updater.update()
                except Exception as exc:
                    self.logger.warning("Failed to update news sentiment for /sentiment: %s", exc)
            lines.append("<b>News</b>")
            lines.append(format_news_sentiment_telegram(news_payload))
        else:
            lines.append("<b>News</b>")
            lines.append("⚪ News: N/A")

        lines.append("")

        # Fear & Greed
        lines.append("<b>Fear & Greed</b>")
        try:
            fg = await SentimentCollector().get_sentiment()
        except Exception as exc:
            self.logger.warning("Failed to fetch Fear & Greed sentiment: %s", exc)
            fg = None

        if fg:
            trend = fg.fear_greed_trend
            lines.append(
                f"🧭 Fear & Greed: {fg.fear_greed_value}/100 ({fg.fear_greed_label}) | trend: {trend} ({fg.fear_greed_change:+d})"
            )
        else:
            lines.append("🧭 Fear & Greed: N/A")

        lines.append("")

        # Reddit sentiment
        lines.append("<b>Reddit</b>")
        reddit = None
        if self.reddit_monitor:
            try:
                reddit = await self.reddit_monitor.fetch_sentiment()
            except Exception as exc:
                self.logger.warning("Failed to fetch Reddit sentiment: %s", exc)

        if reddit:
            top = ", ".join(f"{ticker}({count})" for ticker, count in reddit.top_tickers[:3]) or "none"
            lines.append(
                f"💬 Reddit score: {reddit.sentiment_score:+.1f} | posts: {reddit.posts_analyzed} | top tickers: {top}"
            )
        else:
            lines.append("💬 Reddit: Not configured / unavailable")

        return "\n".join(lines)

    async def _send_market_close_notification(self, now_et: datetime) -> None:
        # Send end-of-day summary when market closes.
        if not self.telegram:
            return

        farm = self.paper_trader_farm
        aggregate = farm.get_aggregate_pnl() if farm else {}
        top_gains = await farm.get_top_unrealized(n=3) if farm else []

        lines = [
            f"🔔 <b>MARKET CLOSE SUMMARY</b> — {now_et.strftime('%b %d, %Y')}",
            "",
            "<b>Today's Result:</b>",
            f"• Realized P&L: <b>${aggregate.get('realized_pnl', 0):+.2f}</b>",
            f"• Win Rate: {aggregate.get('win_rate', 0):.1f}%",
            "",
            "<b>Activity Breakdown:</b>",
            f"• New Entries: {self._today_opened:,}",
            f"• Manual/Exit Closures: {self._today_closed:,}",
            f"• Expirations: {self._today_expired:,}",
            "",
        ]

        if top_gains:
            lines.append("<b>Top Unrealized Runners:</b>")
            for i, g in enumerate(top_gains, 1):
                pnl_emoji = "🟢" if g['unrealized_pnl'] > 0 else "🔴"
                lines.append(
                    f"{i}. {g['symbol']} {g['strategy']} {g['strikes']}\n"
                    f"   {pnl_emoji} <b>${g['unrealized_pnl']:+.2f} ({g['pnl_pct']:+.1f}%)</b>"
                )
            lines.append("")

        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        
        lines.append(
            f"Settlement Score: {conditions.get('score', 'N/A')}/10 "
            f"{str(conditions.get('warmth_label', 'N/A')).upper()}"
        )

        await self.telegram.send_tiered_message("\n".join(lines), priority=2, key="market_close")
    
# Dashboard helper callbacks

    def _get_snapshot_section(self, key: str) -> Optional[Any]:
        # Fetch a cached snapshot section if available.
        return self._status_snapshot.get(key)

    async def _refresh_status_snapshot(self, force: bool = False) -> None:
        # Refresh cached dashboard/telegram status snapshot.
        now = time.time()
        if not force and (now - self._status_snapshot_ts) < self._status_snapshot_interval:
            return

        async with self._status_snapshot_lock:
            now = time.time()
            if not force and (now - self._status_snapshot_ts) < self._status_snapshot_interval:
                return

            started = time.perf_counter()
            snapshot: Dict[str, Any] = {}
            try:
                db_stats = await self.db.get_db_stats()
                snapshot['system'] = {
                    'db_size_mb': db_stats.get('db_size_mb', 0),
                    'boot_phases': self._format_boot_phases(),
                }
                snapshot['db_stats'] = db_stats
            except Exception as e:
                self.logger.warning(f"Status snapshot system error: {e}")
                snapshot['system'] = {'db_size_mb': 0, 'boot_phases': self._format_boot_phases()}
                snapshot['db_stats'] = {'db_size_mb': 0, 'row_counts': {}}

            try:
                snapshot['providers'] = await self._get_provider_statuses()
            except Exception as e:
                self.logger.warning(f"Status snapshot providers error: {e}")
                snapshot['providers'] = {'error': str(e)}
            try:
                snapshot['detectors'] = await self._get_detector_statuses()
            except Exception as e:
                self.logger.warning(f"Status snapshot detectors error: {e}")
                snapshot['detectors'] = {'error': str(e)}
            try:
                if self.query_layer:
                    status_v2 = await self.query_layer.status()
                    snapshot['internal'] = status_v2.get('internal', {})
                    snapshot['bus'] = status_v2.get('bus', {})
                    snapshot['db'] = status_v2.get('db', {})
                else:
                    snapshot['internal'] = {}
                    snapshot['bus'] = {}
                    snapshot['db'] = {}
            except Exception as e:
                self.logger.warning(f"Status snapshot internal error: {e}")
                snapshot['internal'] = {}
                snapshot['bus'] = {}
                snapshot['db'] = {}
            if self.telegram:
                snapshot['internal']['telegram'] = self.telegram.get_status()

            try:
                snapshot['pnl'] = await self._compute_pnl_summary()
            except Exception as e:
                self.logger.warning(f"Status snapshot pnl error: {e}")
                snapshot['pnl'] = {'error': str(e)}

            try:
                snapshot['farm'] = await self._compute_farm_status()
            except Exception as e:
                self.logger.warning(f"Status snapshot farm error: {e}")
                snapshot['farm'] = {'error': str(e)}

            try:
                snapshot['positions'] = await self._compute_positions_summary()
            except Exception as e:
                self.logger.warning(f"Status snapshot positions error: {e}")
                snapshot['positions'] = []

            try:
                snapshot['data_status'] = await self._get_data_status()
            except Exception as e:
                self.logger.warning(f"Status snapshot data_status error: {e}")
                snapshot['data_status'] = {}

            try:
                snapshot['recent_logs'] = await self._read_recent_logs_text()
            except Exception as e:
                self.logger.warning(f"Status snapshot logs error: {e}")
                snapshot['recent_logs'] = f"Error: {e}"

            self._status_snapshot = snapshot
            self._status_snapshot_ts = now

            # Truth feed health (per-asset Coinbase/OKX liveness)
            if hasattr(self, '_truth_feed_health') and self._truth_feed_health:
                snapshot['truth_feed_health'] = dict(self._truth_feed_health)

            elapsed_ms = (time.perf_counter() - started) * 1000
            self.logger.debug(f"Status snapshot refreshed in {elapsed_ms:.1f}ms")

    async def _refresh_zombies_snapshot(self, force: bool = False) -> None:
        # Refresh cached zombies report separately (heavier query).
        now = time.time()
        if not force and (now - self._zombies_snapshot_ts) < self._zombies_snapshot_interval:
            return
        if not self.paper_trader_farm:
            self._zombies_snapshot = {'zombies': [], 'total': 0, 'report': 'Farm not initialized'}
            self._zombies_snapshot_ts = now
            return

        started = time.perf_counter()
        try:
            zombies = await self.paper_trader_farm.detect_zombies(stale_days=14, grace_days=2)
            report = await self.paper_trader_farm.format_zombies_report(stale_days=14, grace_days=2)
            self._zombies_snapshot = {
                'zombies': zombies,
                'total': len(zombies),
                'report': report,
            }
        except Exception as e:
            self.logger.warning(f"Zombies snapshot error: {e}")
            self._zombies_snapshot = {'zombies': [], 'total': 0, 'report': f"Error: {e}"}
        self._zombies_snapshot_ts = now
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.logger.debug(f"Zombies snapshot refreshed in {elapsed_ms:.1f}ms")

    async def _get_dashboard_system_status(self) -> Dict[str, Any]:
        # System status for the dashboard /api/status endpoint.
        cached = self._get_snapshot_section('system')
        if cached:
            extra = {
                "internal": self._get_snapshot_section("internal") or {},
                "bus": self._get_snapshot_section("bus") or {},
                "db": self._get_snapshot_section("db") or {},
                "providers": self._get_snapshot_section("providers") or {},
                "detectors": self._get_snapshot_section("detectors") or {},
            }
            return {**cached, **extra}
        await self._refresh_status_snapshot(force=True)
        system = self._status_snapshot.get('system', {'db_size_mb': 0, 'boot_phases': self._format_boot_phases()})
        extra = {
            "internal": self._status_snapshot.get("internal", {}),
            "bus": self._status_snapshot.get("bus", {}),
            "db": self._status_snapshot.get("db", {}),
            "providers": self._status_snapshot.get("providers", {}),
            "detectors": self._status_snapshot.get("detectors", {}),
        }
        return {**system, **extra}

    async def _get_provider_statuses(self) -> Dict[str, Any]:
        # Provider health for dashboard using the activity tracker.
        return self._activity_tracker.get_provider_statuses()

    def _get_current_risk_flow(self) -> Optional[float]:
        # Get the last computed GlobalRiskFlow value.
        if self.global_risk_flow_updater:
            return self.global_risk_flow_updater._last_value
        return None

    async def _format_briefing_prices(self) -> List[str]:
        # Format SPY, IBIT, QQQ prices for market briefing. Returns list of lines.
        lines = []
        bar_source = (self.config.get("data_sources") or {}).get("bars_primary", "alpaca")
        for symbol in ("SPY", "IBIT", "QQQ"):
            price_str = None
            if symbol == "IBIT":
                det = self.detectors.get("ibit")
                if det and getattr(det, "_current_ibit_data", None):
                    price = det._current_ibit_data.get("price", 0)
                    change = det._current_ibit_data.get("change_pct", 0)
                    if price:
                        emoji = "↗️" if change >= 0 else "↘️"
                        price_str = f"• {symbol}: ${price:.2f} {emoji} {change:+.2f}%"
            else:
                close = await self.db.get_latest_bar_close(bar_source, symbol, 60) if self.db else None
                if close is not None:
                    price_str = f"• {symbol}: ${close:.2f}"
            if price_str:
                lines.append(price_str)
        return lines

    def _format_current_regime_for_telegram(self) -> str:
        # Format current equity regime (vol/trend/risk) for Telegram. Uses last emitted regime (bar-driven).
        if not self.regime_detector:
            return "📊 Regime: N/A"
        try:
            event = self.regime_detector.get_market_regime("EQUITIES")
            if not event:
                return "📊 Regime: N/A (no bars yet)"
            risk = getattr(event, "risk_regime", "UNKNOWN")
            spy_vol = spy_trend = None
            if getattr(event, "metrics_json", None):
                try:
                    import json
                    metrics = json.loads(event.metrics_json)
                    spy = metrics.get("SPY") if isinstance(metrics.get("SPY"), dict) else None
                    if spy:
                        spy_vol = spy.get("vol_regime")
                        spy_trend = spy.get("trend_regime")
                except Exception:
                    pass

            parts = [f"Risk: {risk}"]
            if spy_vol:
                parts.append(f"SPY Vol: {spy_vol}")
            if spy_trend:
                parts.append(f"Trend: {spy_trend}")
            return "📊 Regime: " + " | ".join(parts)
        except Exception as e:
            self.logger.debug("Regime format for telegram: %s", e)
            return "📊 Regime: N/A"

    async def _get_detector_statuses(self) -> Dict[str, Any]:
        # Detector activity status for dashboard.
        return self._activity_tracker.get_detector_statuses()

    async def _compute_pnl_summary(self) -> Dict:
        # Compute P&L summary without cached shortcut.
        if self.paper_trader_farm:
            return await self.paper_trader_farm.get_pnl_for_telegram()
        if self.daily_review:
            from .analysis.daily_review import get_pnl_summary
            return await get_pnl_summary(self.daily_review)
        return {}

    async def _compute_positions_summary(self) -> List[Dict]:
        # Compute positions summary without cached shortcut.
        if self.paper_trader_farm:
            return await self.paper_trader_farm.get_positions_for_telegram()
        return []

    async def _compute_farm_status(self) -> Dict[str, Any]:
        # Compute farm status without cached shortcut.
        if not self.paper_trader_farm:
            return {}
        return self.paper_trader_farm.get_status_summary()

    async def _run_dashboard_command(self, cmd: str) -> str:
        # Execute a / command from the web dashboard.
        cmd = cmd.strip()
        if not cmd.startswith('/'):
            cmd = '/' + cmd

        try:
            if cmd == '/pnl':
                data = await self._compute_pnl_summary()
                return (
                    f"Today: ${data.get('today_pnl', 0):+.2f}\n"
                    f"MTD: ${data.get('month_pnl', 0):+.2f}\n"
                    f"YTD: ${data.get('year_pnl', 0):+.2f}\n"
                    f"Win rate MTD: {data.get('win_rate_mtd', 0):.0f}%\n"
                    f"Open positions: {data.get('open_positions', 0)}"
                )
            elif cmd == '/status':
                if self.query_layer:
                    data = await self.query_layer.status()
                    return str(data)
                return "Query layer not initialized"
            elif cmd == '/positions':
                data = await self._compute_positions_summary()
                return "\n".join(
                    f"{p['symbol']} ({p['strategy']}): {p['count']} positions"
                    for p in data
                ) or "No open positions"
            elif cmd == '/zombies':
                data = self._zombies_snapshot
                return data.get('report', f"Total zombies: {data.get('total', 0)}")
            elif cmd.startswith('/zombie_clean'):
                if self.paper_trader_farm:
                    n = await self.paper_trader_farm.close_zombies()
                    return f"Closed {n} zombie positions"
                return "Farm not initialized"
            elif cmd == '/dashboard':
                data = await self._get_dashboard_system_status()
                return str(data)
            elif cmd.startswith('/reset_paper'):
                parts = cmd.split()
                scope = 'all'
                mode = 'epoch'
                for p in parts[1:]:
                    if p.startswith('--scope='):
                        scope = p.split('=')[1]
                    elif p.startswith('--mode='):
                        mode = p.split('=')[1]
                if self.paper_trader_farm:
                    result = await self.paper_trader_farm.reset_paper_equity(scope=scope, mode=mode)
                    await self._refresh_status_snapshot(force=True)
                    await self._refresh_zombies_snapshot(force=True)
                    return result
                return "Farm not initialized"
            elif cmd == '/db_stats':
                stats = self._get_snapshot_section('db_stats')
                if not stats:
                    await self._refresh_status_snapshot(force=True)
                    stats = self._status_snapshot.get('db_stats', {})
                lines = [f"DB size: {stats['db_size_mb']} MB"]
                for table, count in stats.get('row_counts', {}).items():
                    lines.append(f"  {table}: {count:,} rows")
                return "\n".join(lines)
            elif cmd == '/maintenance':
                stats = await self.db.run_maintenance()
                return f"Maintenance complete. DB: {stats['db_size_mb']} MB"
            elif cmd == '/db':
                if self.query_layer:
                    data = await self.query_layer.db()
                    return str(data)
                return "Query layer not initialized"
            elif cmd.startswith('/sql '):
                query = cmd[5:].strip()
                if not query:
                    return "Usage: /sql [SELECT ...]"
                try:
                    rows = await self.db.fetch_all(query)
                    if not rows:
                        return "No rows returned."
                    
                    # Convert objects to readable strings
                    lines = []
                    headers = rows[0].keys()
                    lines.append(" | ".join(headers))
                    lines.append("-" * 40)
                    for r in rows[:50]:  # Limit to 50 rows for display
                        lines.append(" | ".join(str(r[k]) for k in headers))
                    if len(rows) > 50:
                        lines.append(f"... and {len(rows)-50} more")
                    return "\n".join(lines)
                except Exception as e:
                    return f"SQL Error: {e}"
            else:
                return f"Unknown command: {cmd}"
        except Exception as e:
            return f"Error: {e}"

    async def _get_recent_logs_text(self) -> str:
        # Return recent log lines as text for dashboard.
        cached = self._get_snapshot_section('recent_logs')
        if cached:
            return cached
        await self._refresh_status_snapshot(force=True)
        cached = self._status_snapshot.get('recent_logs')
        if cached:
            return cached
        return await self._read_recent_logs_text()

    async def _read_recent_logs_text(self) -> str:
        # Read recent log lines directly (no cache).
        if self._recent_logs:
            return "\n".join(self._recent_logs)
        # Read from log file as fallback
        try:
            log_file = Path(self.config.get('logging', {}).get('log_dir', 'data/logs')) / 'argus.log'
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                return "".join(lines[-100:])
        except Exception:
            pass

        return "No logs available"

# Market hours gating

    def _is_us_market_open(self) -> bool:
        # Check if US equity markets are open (Mon-Fri, 9:30-16:00 ET).
        eastern = ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
        if now_et.weekday() >= 5:  # Weekend
            return False
        mh = self._mh_cfg
        open_h = int(mh.get('equity_open_hour', 9))
        open_m = int(mh.get('equity_open_minute', 30))
        close_h = int(mh.get('equity_close_hour', 16))
        close_m = int(mh.get('equity_close_minute', 0))
        open_time = now_et.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        close_time = now_et.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        return open_time <= now_et <= close_time

# DB maintenance task

    async def _run_db_maintenance(self) -> None:
        # Periodic DB maintenance: retention cleanup + PRAGMA optimize.
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Retention cleanup
                retention = self.config.get('data_retention', {})
                retention_map = {
                    'price_snapshots': retention.get('price_snapshots_days', 30),
                    'system_health': retention.get('logs_days', 30),
                    'funding_rates': retention.get('price_snapshots_days', 30),
                    'liquidations': retention.get('price_snapshots_days', 30),
                    'options_iv': retention.get('detections_days', 180),
                    'detections': retention.get('detections_days', 180),
                    'option_chain_snapshots': retention.get('option_chain_snapshots_days', 30),
                }
                await self.db.cleanup_old_data(retention_map)

                # PRAGMA optimize
                await self.db.run_maintenance()
                self.logger.info("DB maintenance completed")
            except Exception as e:
                self.logger.error(f"DB maintenance error: {e}")

    # ── Soak-test hardening helpers ─────────────────────────────────

    async def _get_soak_summary(self) -> Dict[str, Any]:
        # Build the soak summary for /debug/soak endpoint.
        return build_soak_summary(
            bus=self.event_bus,
            bar_builder=self.bar_builder,
            persistence=self.persistence,
            feature_builder=self.feature_builder,
            regime_detector=self.regime_detector,
            resource_monitor=self.resource_monitor,
            guardian=self.soak_guardian,
            tape_recorder=self.tape_recorder,
            providers=self._activity_tracker.get_provider_statuses(),
            detectors=self._activity_tracker.get_detector_statuses(),
            polymarket_gamma=getattr(self, 'polymarket_gamma', None),
            polymarket_clob=getattr(self, 'polymarket_clob', None),
            polymarket_watchlist=getattr(self, 'polymarket_watchlist', None),
            bybit_ws=self.bybit_ws,
        )

    async def _send_soak_alert(
        self, severity: str, guard: str, message: str
    ) -> None:
        # Send a soak guard alert via Telegram (rate-limited by guardian).
        if not self.telegram:
            return
        icon = "\u26a0\ufe0f" if severity == "WARN" else "\u274c"
        text = (
            f"{icon} <b>Soak Guard: {guard}</b>\n"
            f"Severity: {severity}\n"
            f"{message}"
        )
        try:
            await self.telegram.send_message(text)
        except Exception as e:
            self.logger.warning(f"Failed to send soak alert: {e}")

    async def _export_tape(self, last_n_minutes: Optional[int] = None) -> List[Dict[str, Any]]:
        # Extract a slice of recorded tape data for export.
        if not self.tape_recorder:
            return []
        with self.tape_recorder._lock:
            snapshot = list(self.tape_recorder._tape)
        
        if not last_n_minutes:
            return snapshot
            
        now = time.time()
        cutoff = now - (last_n_minutes * 60)
        return [e for e in snapshot if e.get("timestamp", 0) >= cutoff]

    async def _run_soak_guards(self) -> None:
        # Periodically evaluate soak guards (every 30s).
        interval = int(
            self.config.get('soak', {}).get('guard_interval_s', 30)
        )
        while self._running:
            await asyncio.sleep(interval)
            try:
                bus_stats = self.event_bus.get_status_summary()
                bb_status = self.bar_builder.get_status() if self.bar_builder else {}
                persist_status = self.persistence.get_status() if self.persistence else {}
                resource_snap = self.resource_monitor.get_full_snapshot()

                self.soak_guardian.evaluate(
                    bus_stats=bus_stats,
                    bar_builder_status=bb_status,
                    persistence_status=persist_status,
                    resource_snapshot=resource_snap,
                    component_heartbeats=self._component_heartbeat_ts,
                )
            except Exception:
                self.logger.debug("Soak guard evaluation failed", exc_info=True)

    async def run(self) -> None:
        # Start all components and run main loop.
        self._running = True
        is_collector = self.mode == "collector"

        self.logger.info("Starting Argus (mode=%s)...", self.mode)

        # Start WebSocket connections
        # NOTE: Bybit is auxiliary analytics only — NOT truth-critical.
        # Coinbase WS is the PRIMARY truth source for crypto price data.
        if self.bybit_ws:
            self._tasks.append(asyncio.create_task(self.bybit_ws.connect()))

        # ── Coinbase WS: primary truth feed (sub-second crypto ticks) ──
        if self.coinbase_ws:
            self._tasks.append(asyncio.create_task(self.coinbase_ws.connect()))
            self.logger.info("Coinbase WS truth feed started (primary)")

        # ── OKX WS: secondary fallback (fires when Coinbase silent > %.1fs) ──
        if self._okx_fallback:
            await self._okx_fallback.start()
            self.logger.info("OKX WS fallback feed started (secondary)")

        # Start polling tasks (with market-hours gating for equities)
        if self.yahoo_client:
            self._tasks.append(asyncio.create_task(self._poll_yahoo_market_hours_aware()))

        # Alpaca bars polling (runs in ALL modes - data only, no execution)
        # Uses FIXED INTERVAL polling - no market-hours gating for determinism
        if self.alpaca_client:
            self._tasks.append(asyncio.create_task(self.alpaca_client.poll()))

        # Alpaca OPTIONS chain polling (Phase 3B - runs in ALL modes)
        if self.alpaca_options:
            self._tasks.append(asyncio.create_task(self._poll_options_chains()))

        # Tastytrade OPTIONS chain polling (runs in ALL modes, independent of Alpaca)
        if self.tastytrade_options:
            self._tasks.append(asyncio.create_task(self._poll_tastytrade_options_chains()))
            # DXLink Greeks require option-level symbols (not underlyings); fetch once at start
            try:
                dxlink_symbols = await asyncio.to_thread(
                    self.tastytrade_options.get_dxlink_option_symbols,
                    getattr(self, "_tastytrade_options_symbols", []),
                    getattr(self, "_tastytrade_options_min_dte", 7),
                    getattr(self, "_tastytrade_options_max_dte", 21),
                    10_000,  # dxFeed allows 100k; distribute equally among equities for IV enrichment
                )
                self._dxlink_greeks_symbols = dxlink_symbols if dxlink_symbols else getattr(self, "_tastytrade_options_symbols", [])
                if dxlink_symbols:
                    self.logger.info(
                        "DXLink Greeks will subscribe to %d option symbols (IV enrichment)",
                        len(dxlink_symbols),
                    )
                else:
                    self.logger.warning(
                        "DXLink Greeks: no option symbols from chain fetch; subscribing to underlyings only (Greeks may be empty)"
                    )
            except Exception as exc:
                self.logger.warning("DXLink option symbol fetch failed, using underlyings: %s", exc)
                self._dxlink_greeks_symbols = getattr(self, "_tastytrade_options_symbols", [])
            # DXLink Greeks streamer populates _greeks_cache for IV enrichment
            self._tasks.append(asyncio.create_task(self._start_dxlink_greeks_streamer()))
            self._tasks.append(asyncio.create_task(self._log_iv_status()))

        # Public OPTIONS chain polling (provider=public; snapshots already contain IV from Public greeks)
        if self.public_options:
            self._tasks.append(asyncio.create_task(self._poll_public_options_chains()))

        self._tasks.append(asyncio.create_task(self._poll_deribit()))
        self._tasks.append(asyncio.create_task(self._health_check()))
        self._tasks.append(asyncio.create_task(self._run_db_maintenance()))

        # Soak guard evaluation loop
        self._tasks.append(asyncio.create_task(self._run_soak_guards()))

        # Heartbeat publisher (drives persistence flush boundaries)
        self._tasks.append(asyncio.create_task(self._publish_heartbeats()))
        self._tasks.append(asyncio.create_task(self._publish_minute_ticks()))

        # ── Trading tasks: DISABLED in collector mode ───
        if not is_collector:
            # Exit monitor runs independently of research loop
            if self.paper_trader_farm:
                self._tasks.append(asyncio.create_task(self._run_exit_monitor()))

            if self.research_enabled:
                self._tasks.append(asyncio.create_task(self._run_research_farm()))
        else:
            self.logger.info("Collector mode: exit monitor and research farm SKIPPED")

        # Market session monitor (open/close notifications)
        self._tasks.append(asyncio.create_task(self._run_market_session_monitor()))

        # Component heartbeat loop (Stream 2)
        self._tasks.append(asyncio.create_task(self._publish_component_heartbeats()))

        # Periodic status snapshots (Stream 2)
        self._tasks.append(asyncio.create_task(self._publish_status_snapshots()))

        # Polymarket polling loops (Stream 3)
        if self.polymarket_gamma:
            self._tasks.append(asyncio.create_task(self.polymarket_gamma.poll_loop()))
        if self.polymarket_clob:
            self._tasks.append(asyncio.create_task(self.polymarket_clob.poll_loop()))
        if self.polymarket_watchlist:
            self._tasks.append(asyncio.create_task(self.polymarket_watchlist.sync_loop()))

        # Automate gap risk snapshots at market close
        if self.gap_risk_tracker:
            self._tasks.append(asyncio.create_task(self._run_market_close_snapshot()))

        # Start AlphaVantage collector loop
        if self.av_collector:
            self._tasks.append(asyncio.create_task(self._run_av_collector_loop()))

        # Start GlobalRiskFlow updater loop
        if self.global_risk_flow_updater:
            self._tasks.append(asyncio.create_task(self._run_external_metrics_loop()))

        # Start conditions monitoring (synthesis layer)
        if self.conditions_monitor:
            self._tasks.append(asyncio.create_task(self.conditions_monitor.start_monitoring()))

        # Start daily review monitoring (4 PM summary)
        if self.daily_review:
            self._tasks.append(asyncio.create_task(self.daily_review.start_monitoring()))

        # Start Telegram two-way polling
        if self.telegram:
            await self.telegram.start_polling()

        self.logger.info("Argus is running! Press Ctrl+C to stop.")

        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            self.logger.info("Shutdown requested")
    
    async def _poll_yahoo_market_hours_aware(self) -> None:
        # Poll Yahoo Finance with market-hours awareness.
        #
        # During US market hours: poll every 60s.
        # Off-hours: poll every off_hours_sample_interval_seconds (default 600s / 10 min).
        off_interval = int(self._mh_cfg.get('off_hours_sample_interval_seconds', 600))
        on_interval = 60

        while self._running:
            try:
                market_open = self._is_us_market_open()
                interval = on_interval if market_open else off_interval

                # One poll cycle
                await self.yahoo_client.poll_once()

                if not market_open:
                    self.logger.debug(f"US market closed, next Yahoo poll in {interval}s")

                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(
                    "Yahoo market-hours poll error (%s): %s",
                    type(e).__name__, e,
                )
                self.logger.debug("Yahoo market-hours poll error detail", exc_info=True)
                await asyncio.sleep(60)

    async def _poll_alpaca_market_hours_aware(self) -> None:
        # Poll Alpaca for equity bars with market-hours awareness.
        #
        # During US market hours: poll every poll_interval (config, default 60s).
        # Off-hours: poll every off_hours_sample_interval_seconds (default 600s).
        #
        # NOTE: Runs in ALL modes (collector, paper, live) - this is DATA only, no execution.
        if not self.alpaca_client:
            return
            
        alpaca_cfg = self.config.get('exchanges', {}).get('alpaca', {})
        on_interval = int(alpaca_cfg.get('poll_interval_seconds', 60))
        off_interval = int(self._mh_cfg.get('off_hours_sample_interval_seconds', 600))

        while self._running:
            try:
                market_open = self._is_us_market_open()
                interval = on_interval if market_open else off_interval

                # One poll cycle
                await self.alpaca_client.poll_once()

                if not market_open:
                    self.logger.debug(f"US market closed, next Alpaca poll in {interval}s")

                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(
                    "Alpaca market-hours poll error (%s): %s",
                    type(e).__name__, e,
                )
                self.logger.debug("Alpaca market-hours poll error detail", exc_info=True)
                await asyncio.sleep(60)

    # ── Heartbeat publisher ───────────────────────────────

    async def _publish_heartbeats(self) -> None:
        # Emit :class:`HeartbeatEvent` every 60 s to drive persistence flushes.
        seq = 0
        interval = self.config.get('monitoring', {}).get('heartbeat_interval', 60)
        while self._running:
            await asyncio.sleep(interval)
            seq += 1
            self.event_bus.publish(
                TOPIC_SYSTEM_HEARTBEAT,
                HeartbeatEvent(sequence=seq),
            )

    async def _publish_component_heartbeats(self) -> None:
        # Emit structured heartbeats for all components every 60s.
        interval = self.config.get('monitoring', {}).get('heartbeat_interval', 60)
        while self._running:
            await asyncio.sleep(interval)
            try:
                if self.bar_builder:
                    self.bar_builder.emit_heartbeat()
                if self.persistence:
                    self.persistence.emit_heartbeat()
                if self.feature_builder:
                    self.feature_builder.emit_heartbeat()
                if self.regime_detector:
                    self.regime_detector.emit_heartbeat()
                # ── System-level heartbeat for uptime tracking (Phase 4A.1+) ──
                if self.db:
                    import time as _t
                    ts_ms = int(_t.time() * 1000)
                    await self.db.write_heartbeat("orchestrator", ts_ms)
            except Exception:
                self.logger.debug("Component heartbeat emission failed", exc_info=True)

    async def _publish_status_snapshots(self) -> None:
        # Periodically persist QueryLayer status snapshots to DB.
        interval = self.config.get('monitoring', {}).get('status_snapshot_persist_interval', 300)
        while self._running:
            await asyncio.sleep(interval)
            try:
                if self.query_layer:
                    await self.query_layer.persist_snapshot()
            except Exception:
                self.logger.debug("Status snapshot persist failed", exc_info=True)

    async def _publish_minute_ticks(self) -> None:
        # Emit minute-boundary ticks aligned to UTC minute boundaries.
        while self._running:
            now = time.time()
            next_minute = (int(now // 60) + 1) * 60
            await asyncio.sleep(max(0.0, next_minute - now))
            if not self._running:
                break
            self.event_bus.publish(
                TOPIC_SYSTEM_MINUTE_TICK,
                MinuteTickEvent(timestamp=next_minute),
            )

    async def _run_av_collector_loop(self) -> None:
        # Periodic loop for continuous Alpha Vantage polling.
        if self.av_collector:
            try:
                await self.av_collector.run_forever()
            except Exception:
                self.logger.error("AlphaVantageCollector loop crash", exc_info=True)

    async def _run_external_metrics_loop(self) -> None:
        # Periodic loop to compute and publish external metrics.
        #
        # Supports global_risk_flow and news_sentiment with independent
        # intervals. The loop is non-blocking across metric updaters.
        if not self.global_risk_flow_updater and not self.news_sentiment_updater:
            return

        av_cfg = self.config.get('exchanges', {}).get('alphavantage', {})
        ns_cfg = self.config.get('news_sentiment', {})
        grf_interval = int(av_cfg.get('external_metrics_interval_seconds', 3600))
        ns_interval = int(ns_cfg.get('interval_seconds', 3600))
        ns_enabled = bool(ns_cfg.get('enabled', False))

        base_sleep = max(1, min(grf_interval, ns_interval if ns_enabled else grf_interval))
        next_grf_run = 0.0
        next_ns_run = 0.0

        self.logger.info(
            "External metrics loop started — global_risk_flow=%ds, news_sentiment=%s(%ds)",
            grf_interval,
            "enabled" if ns_enabled else "disabled",
            ns_interval,
        )

        while self._running:
            now = time.time()
            try:
                if self.global_risk_flow_updater and now >= next_grf_run:
                    await self.global_risk_flow_updater.update()
                    next_grf_run = now + grf_interval

                if self.news_sentiment_updater and ns_enabled and now >= next_ns_run:
                    await self.news_sentiment_updater.update()
                    next_ns_run = now + ns_interval
            except Exception:
                self.logger.error("External metrics update failed (will retry)", exc_info=True)

            # Wait for next interval or shutdown
            try:
                await asyncio.sleep(base_sleep)
            except asyncio.CancelledError:
                break

    async def _get_kalshi_summary(self) -> Dict[str, Any]:
        # Fetch summary of Kalshi performance and recent events from sidecar DB.
        try:
            # 1. PnL Stats
            stats = await self.kalshi_db.get_kalshi_outcome_stats()
            
            # 2. Recent Events
            cursor = await self.kalshi_db._connection.execute(
                "SELECT timestamp, level, message FROM kalshi_terminal_events "
                "ORDER BY id DESC LIMIT 5"
            )
            rows = await cursor.fetchall()
            events = [
                {"timestamp": r[0], "level": r[1], "message": r[2]}
                for r in rows
            ]
            
            return {
                "total_pnl": stats.get("total_pnl", 0.0),
                "wins": stats.get("wins", 0),
                "total_trades": stats.get("total", 0),
                "win_rate": (stats.get("wins", 0) / stats.get("total", 1) * 100) if stats.get("total", 0) > 0 else 0.0,
                "recent_events": events
            }
        except Exception as e:
            self.logger.warning(f"Error fetching Kalshi summary: {e}")
            return {"error": str(e)}

    async def stop(self) -> None:
        # Stop all components gracefully.
        self.logger.info("Stopping Argus...")
        self._running = False

        # Flush bar builder partial bars → bus → persistence
        if self.bar_builder:
            self.bar_builder.flush()

        # Flush persistence buffer (writes remaining bars to DB)
        if self.persistence:
            self.persistence.shutdown()

        # Stop event bus workers
        self.event_bus.stop()

        # Stop AlphaVantage Collector
        if self.av_collector:
            self.av_collector.stop()

        # Stop dashboard
        if self.dashboard:
            await self.dashboard.stop()

        if self.news_sentiment_updater:
            await self.news_sentiment_updater.close()

        # Stop monitoring loops
        if self.conditions_monitor:
            self.conditions_monitor.stop_monitoring()
        if self.daily_review:
            self.daily_review.stop_monitoring()

        # Stop Telegram polling
        if self.telegram:
            await self.telegram.stop_polling()

        # Cancel all tasks and await their completion
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop Polymarket clients
        if self.polymarket_gamma:
            await self.polymarket_gamma.stop()
        if self.polymarket_clob:
            await self.polymarket_clob.stop()
        if self.polymarket_watchlist:
            await self.polymarket_watchlist.stop()

        # Disconnect WebSockets
        if self.bybit_ws:
            await self.bybit_ws.disconnect()

        # Disconnect Coinbase WS (truth feed primary)
        if self.coinbase_ws:
            await self.coinbase_ws.disconnect()
            self.logger.info("Coinbase WS disconnected")

        # Stop OKX WS fallback
        if self._okx_fallback:
            await self._okx_fallback.stop()
            self.logger.info("OKX WS fallback stopped")

        # Close REST clients
        if self.deribit_client:
            await self.deribit_client.close()
        if self.yahoo_client:
            await self.yahoo_client.close()
        if self.alpaca_options:
            await self.alpaca_options.close()
        if self.tastytrade_options:
            self.tastytrade_options.close()
        if self.public_options:
            await self.public_options.close()
        if self.alphavantage_client:
            await self.alphavantage_client.close()

        # Stop AI agent (LLM shutdown)
        if hasattr(self, 'ai_agent') and self.ai_agent:
            await self.ai_agent.stop()

        # Close database
        await self.db.close()

        # Send shutdown notification
        if self.telegram:
            await self.telegram.send_system_status('offline', 'Argus stopped')

        self.logger.info("Argus stopped")


async def main() -> None:
    # Entry point for Argus.
    argus = ArgusOrchestrator()
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(argus.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        await argus.setup()
        await argus.run()
    except KeyboardInterrupt:
        pass

    finally:
        await argus.stop()


if __name__ == "__main__":
    asyncio.run(main())
