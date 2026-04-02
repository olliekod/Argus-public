# Created by Oliver Meihls

# Configuration for the Argus Kalshi module.
#
# Reads a structured dict (injected by the Argus config loader) and exposes
# typed, validated settings.  Secrets are loaded separately by Argus — this
# module never touches env-vars or files for credentials directly.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True, slots=True)
class TruthFeedConfig:
    asset: str = "BTC"
    topic: str = "btc.mid_price"
    coinbase_symbol: str = "BTC/USDT"
    publish_to_core_bus: bool = True



@dataclass(frozen=True, slots=True)
class KalshiConfig:
    # Immutable, validated configuration for one Kalshi trading session.

    # ── Bot identity ────────────────────────────────────────────────────
    bot_id: str = "default"

    # ── API endpoints ──────────────────────────────────────────────────
    base_url_rest: str = "https://api.elections.kalshi.com/trade-api/v2"
    base_url_ws: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    # ── Credentials (paths / IDs — actual secrets loaded by Argus) ─────
    kalshi_key_id: str = ""
    kalshi_private_key_path: str = ""

    # ── Rate limits (requests per second) ──────────────────────────────
    rate_limit_read_per_sec: float = 10.0
    rate_limit_write_per_sec: float = 5.0

    # ── Market targeting ───────────────────────────────────────────────
    target_market_tickers: List[str] = field(default_factory=list)
    series_filter: Optional[str] = None
    event_filter: Optional[str] = None
    market_title_regex: str = ""
    market_refresh_interval_s: float = 60.0

    # ── Risk parameters ────────────────────────────────────────────────
    bankroll_usd: float = 1000.0
    max_fraction_per_market: float = 0.10
    daily_drawdown_limit: float = 0.05
    min_edge_threshold: float = 0.03
    # ── Dynamic Risk & Sizing ──────────────────────────────────────────
    # Fraction of current balance to risk per signal (e.g. 0.001 = 0.1%).
    risk_fraction_per_trade: float = 0.001
    sizing_risk_fraction: float = 0.001
    # Halt trading if daily PnL is negative for this many consecutive days.
    consecutive_drawdown_limit: int = 3
    # Use real-time account balance from the exchange for sizing/risk.
    # If False, falls back to static bankroll_usd.
    use_live_balance: bool = True

    # ── Edge calibration (Phase 1) ──────────────────────────────────
    # Fee subtracted from raw EV before comparing to min_edge_threshold.
    #
    # Actual Kalshi fee formula (taker):
    #   fee_dollars = 0.07 × contracts × P × (1 − P)
    # where P is the contract price in dollars (0.01–0.99).
    #
    # This peaks at 50¢ ATM contracts: 0.07 × 0.50 × 0.50 = 0.0175 ($0.0175/contract)
    # and falls toward zero at extreme prices (e.g. 5¢: 0.07×0.05×0.95 = 0.0033).
    #
    # For ONE-WAY trades (hold to settlement): effective fee pct = 0.07 × P × (1-P).
    # The flat 1.75% (0.0175) is the ATM worst-case — conservative for mid-price
    # contracts, but over-estimates fees for extreme strikes.
    #
    # Maker fee = 0.0175 × P × (1-P) = ¼ of taker.  We always use taker pricing
    # (aggressive fills), so 0.0175 is the right calibration.
    #
    # SCALPER NOTE: early exits require TWO fills (buy + sell), so round-trip
    # taker fees = 2 × 0.07 × P × (1-P) = up to 3.5¢ at ATM.  The scalper's
    # profit target (scalp_min_profit_cents) must exceed this to be net-positive.
    #
    # Why 2% (0.02) instead of 1.75% (0.0175):
    # Kalshi rounds the fee UP to the nearest cent.  For any contract priced
    # between ~27–73¢, ceil(1.75¢) = 2¢ — so the actual fee paid is 2¢, not
    # 1.75¢.  Using 2% accurately reflects this rounding and adds a negligible
    # slippage buffer.  The strategy still generates the same signals; it just
    # requires a marginally larger raw edge before firing.
    effective_edge_fee_pct: float = 0.02
    # Same side must be the best EV winner for this many ms before a
    # signal is emitted.  Filters noise and spoofing flicker.
    persistence_window_ms: int = 60
    # Do not send an order if the last truth-feed tick is older than
    # this (ms).  Prevents trading on stale price data.
    latency_circuit_breaker_ms: int = 20

    # ── Flash-crash / violent-move detection ────────────────────────
    # If the BTC price moves by at least this fraction in a single tick,
    # the probability engine bypasses the 250ms throttle and recomputes
    # immediately.  Lets us catch Kalshi's stale orderbooks in ~10ms
    # instead of up to 250ms.  0.005 = 0.5%.
    urgent_move_pct: float = 0.005

    # ── Near-expiry gamma strategy ───────────────────────────────────
    # In the final N minutes of a contract, probability conviction is
    # very high (small price moves have huge probability impact).
    # We relax the persistence and edge requirements to capture these
    # high-confidence, time-sensitive opportunities.
    near_expiry_minutes: int = 5
    near_expiry_min_edge: float = 0.005   # much lower — model is very confident
    near_expiry_persistence_ms: int = 0   # no persistence delay near expiry

    # ── Early-entry block ────────────────────────────────────────────
    # Refuse to open a new position if the contract still has MORE than
    # this many minutes until settlement.  0 = disabled (any time OK).
    #
    # Rationale: a 60-min contract with 50 min remaining has enormous
    # uncertainty — BTC can move ±3% in that time and flip any edge.
    # 15-min contracts are always < 15 min away so they always pass
    # a limit of, say, 30.  60-min contracts only enter in the last
    # 30 min (the final third of the window, where conviction is higher).
    max_entry_minutes_to_expiry: int = 0

    # ── Minimum time-to-expiry guard (hold strategy) ────────────────────
    # Block mispricing_hold entries when time-to-settlement is LESS than
    # this value.  Prevents the near-expiry sigma collapse: with < 5 min
    # remaining the Gaussian σ < 0.2%, making p_yes hyper-sensitive to
    # any tiny price gap and producing extreme (near 0 or 1) probabilities.
    # For 15m contracts the hold entry window is already only 0–3 min, so
    # setting this to 5 effectively disables 15m mispricing_hold entirely.
    # For 60m contracts (12-min window), valid window becomes 5–12 min.
    # 0 = disabled (legacy behaviour, no lower bound).
    hold_min_entry_minutes_to_expiry: int = 0

    # ── Range-market expiry cap ──────────────────────────────────────
    # For RANGE contracts (e.g. "BTC between 68k–69k"), refuse entry if
    # time to settlement exceeds this many minutes.  Range markets can
    # expire 22+ hours out; trading those is risky (too much can go wrong).
    # Default 60 = only trade range markets expiring within the hour.
    # 0 = disabled (no cap).
    range_max_entry_minutes_to_expiry: int = 60

    # ── Signal cooldown ─────────────────────────────────────────────────
    # After emitting a signal for (ticker, side), suppress further signals
    # for that pair for this many seconds. Prevents fill spam and lets the
    # execution engine process the first fill before reconsidering the market.
    signal_cooldown_s: float = 30.0

    # ── Position accumulation guard ─────────────────────────────────────
    # Hard cap on total contracts held per ticker across all fills.
    # Once a ticker accumulates this many contracts (absolute), no more
    # entries are allowed until settlement clears it.
    # 0 = no cap (legacy behaviour). Recommended: 2–5 conservative, 10–80 aggressive.
    max_contracts_per_ticker: int = 0

    # ── Base contract quantity (conviction multiplier) ───────────────────
    # Minimum contracts to buy per signal regardless of edge.  The actual
    # quantity scales with edge: qty = base_contract_qty × edge_multiples.
    # 1 = one contract per min-edge step (conservative default).
    # 10 = ten contracts per step; e.g. 2% edge → 10, 4% → 20, 16% → 80.
    # Still capped by max_contracts_per_ticker and USD exposure limit.
    base_contract_qty: int = 1

    # ── Absolute entry price limits ─────────────────────────────────────
    # Skip any signal if the ask price is below min_entry_cents or above
    # max_entry_cents.  The primary use-case is blocking 1–5¢ contracts
    # where Kalshi's fee rounds up to 1¢ = 100% of the contract price
    # (you spend 2¢ all-in to potentially win 99¢ — the break-even
    # probability of ~2% is rarely achieved in practice).
    # 0/100 = disabled.
    min_entry_cents: int = 0
    max_entry_cents: int = 100
    no_avoid_above_cents: int = 0
    hold_tail_penalty_start_cents: int = 0
    hold_tail_penalty_per_10c: float = 0.0
    # Hold sleeve must clear a strict net edge gate after one-way entry
    # friction (fee + modeled entry slippage + explicit buffer).
    hold_min_net_edge_cents: int = 1
    hold_entry_cost_buffer_cents: int = 1
    # Hold entry uses market-implied divergence by default:
    # divergence = adjusted_truth_p_yes - market_p_yes.
    # Entry requires |divergence| >= threshold (subject to depth/delta tiebreak).
    hold_min_divergence_threshold: float = 0.0
    # Directional agreement gates for hold entries (enabled by default).
    hold_require_momentum_agreement: bool = True
    hold_require_flow_agreement: bool = True
    # Minimum magnitude thresholds for agreement gates.  Values below these are
    # treated as noise (no clear signal) and the entry is blocked, preventing
    # near-zero drift/flow from passing as spurious agreement.
    # 0.0 = disabled (legacy behaviour: any nonzero value is accepted).
    hold_momentum_agreement_min_drift: float = 0.0
    hold_flow_agreement_min_flow: float = 0.0
    # Reversal trigger for early-exit hold logic.
    hold_flow_reversal_threshold: float = 0.3

    # ── OBI directional bias ─────────────────────────────────────────────
    # Adjusts p_yes by (ob.obi * obi_p_yes_bias_weight) before edge calc.
    # ob.obi > 0 = more YES bid depth (Kalshi market leaning bullish).
    # ob.obi < 0 = more NO bid depth (Kalshi market leaning bearish).
    # This makes the probability estimate directionally aware without
    # hard-suppressing any side — contracts only get passed when the
    # adjusted edge genuinely falls below the min_edge threshold.
    # 0.0 = disabled (default). Calibrate from run data; suggest 0.10–0.20
    # once you have enough settlements to measure the OBI-vs-outcome corr.
    obi_p_yes_bias_weight: float = 0.0

    # ── Momentum directional bias ────────────────────────────────────────────
    # Scales the per-second log-drift (from 30s OLS slope of truth-feed prices)
    # into an additive p_yes adjustment before edge calculation.
    # drift > 0 (uptrend) → pushes p_yes up; drift < 0 → pushes p_yes down.
    # Weight is applied as: p_yes += drift * weight
    # where drift is per-second log-drift (~0.0001/s typical).
    # Typical range: 0.0 (off) to 500.0 (aggressive). Default 0 = disabled.
    momentum_p_yes_bias_weight: float = 0.0

    # ── Trade tape flow bias ─────────────────────────────────────────────────
    # Scales the 60-second YES/NO taker flow imbalance per ticker into an
    # additive p_yes adjustment before edge calculation.
    # flow ∈ [-1, +1]: +1 = all YES takers, -1 = all NO takers.
    # p_yes += flow * weight. Typical range: 0.0 (off) to 0.15.
    # Default 0 = disabled (no trade tape data yet).
    trade_flow_p_yes_bias_weight: float = 0.0

    # ── YES mid-range price filter ──────────────────────────────────────
    # Skip YES signals when the ask price is in [yes_avoid_min_cents,
    # yes_avoid_max_cents].  Backtesting shows at-the-money YES contracts
    # (≈40–62¢) have sub-50% win rates in this strategy.
    # Set both to 0 to disable.
    yes_avoid_min_cents: int = 0
    yes_avoid_max_cents: int = 0

    # ── Mispricing scalper ───────────────────────────────────────────────
    # When False, the MispricingScalper is not started.  Set to False if
    # the scalper is net-negative to isolate edge to the main strategy.
    scalper_enabled: bool = True
    # Minimum profit to take an early exit (cents).  The scalper sells
    # when the current best bid is at least entry + scalp_min_profit_cents.
    scalp_min_profit_cents: int = 2
    # Stop-loss (cents).  If the current bid drops this many cents below
    # the entry price, sell immediately to cap the loss.  Prevents a
    # losing scalp position from riding to settlement where the full
    # entry price is at risk.  0 = no stop-loss.  No closed-form formula;
    # use the farm to test several values in parallel (e.g. 0, 1, 2, 3 or
    # 4–8¢) and compare PnL by band.
    scalp_stop_loss_cents: int = 0
    # Force-exit any scalp position that hasn't hit the profit target
    # within this many minutes (to avoid holding through settlement).
    scalp_max_hold_minutes: float = 10.0
    # Require the fair value for the chosen side to have moved by at least this
    # many cents within scalp_reprice_window_s. This keeps the scalper focused
    # on fast Kalshi catch-up opportunities rather than static conviction.
    scalp_min_reprice_move_cents: int = 4
    # Maximum age of the repricing impulse used to justify a new scalp entry.
    scalp_reprice_window_s: float = 3.0
    # Minimum projected net edge buffer (cents) beyond the raw profit target to
    # cover round-trip slippage/spread drag before we allow a scalp entry.
    scalp_entry_cost_buffer_cents: int = 4
    # After this many seconds, allow a profitable exit when the modeled edge has
    # mostly collapsed and Kalshi appears to have caught up.
    scalp_exit_grace_s: float = 3.0
    # Symmetric stop-loss: exit immediately when bid drops this many cents below
    # entry. 0 = disabled. Set to scalp_min_profit_cents for symmetric risk/reward.
    scalp_stop_loss_cents: int = 0
    # A profitable scalp can be exited once remaining modeled edge for the held
    # side is at or below this threshold (cents).
    scalp_exit_edge_threshold_cents: int = 1
    # Entry price range filter.  Only enter a scalp if the ask is within
    # [scalp_min_entry_cents, scalp_max_entry_cents].  Extreme contracts
    # (>85¢ or <15¢) have asymmetric risk: you risk the full entry price
    # to collect 2¢.  0/100 = no filter (legacy behaviour).
    scalp_min_entry_cents: int = 0
    scalp_max_entry_cents: int = 100
    # Maximum contracts per scalp entry signal.  The actual quantity scales
    # with edge: quantity = min(scalp_max_quantity, edge_cents // 5).
    # e.g. edge=5c→1, edge=10c→2, edge=15c→3 (capped here).
    # Still subject to the execution engine's max_contracts_per_ticker cap.
    scalp_max_quantity: int = 3
    # Minimum edge required to enter a scalp (cents).
    # Reverting to 5c (was 3c) to reduce toxic entries and ensure
    # sufficient profit after fees and slippage.
    scalp_min_edge_cents: int = 5
    # Maximum allowed round-trip spread (YES_ask - YES_bid + NO_ask - NO_bid).
    # Tightening to 2c (was 4c) to ensure scalp exits are realistic at 5c profit.
    scalp_max_spread_cents: int = 2
    # Minimum seconds between buy signals for the same ticker.
    scalp_cooldown_s: float = 5.0
    scalp_momentum_min_drift: float = 0.00005
    # Directional composite score (default scalp entry model).
    scalp_directional_score_threshold: float = 0.3
    scalp_directional_drift_weight: float = 0.30
    scalp_directional_drift_scale: float = 0.0002
    scalp_directional_obi_weight: float = 0.15
    scalp_directional_flow_weight: float = 0.10
    scalp_directional_depth_weight: float = 0.25
    scalp_directional_delta_yes_weight: float = 0.10
    scalp_directional_delta_no_weight: float = 0.10
    # Momentum/reprice sleeve (wallet-style fast reaction) that can enter
    # even when static fair-vs-ask mispricing is weak, provided short-horizon
    # repricing and microstructure confirmation are present.
    momentum_scalp_enabled: bool = True
    momentum_min_reprice_move_cents: int = 3
    momentum_reprice_window_s: float = 4.0
    momentum_min_orderbook_imbalance: float = 0.30
    momentum_max_spread_cents: int = 3
    momentum_min_edge_cents: int = 0
    momentum_min_profit_cents: int = 2
    momentum_entry_cost_buffer_cents: int = 5
    momentum_max_quantity: int = 2
    # Pair arbitrage sleeve: buy both YES and NO only when summed asks are
    # sufficiently below 100c after estimated fees/slippage.
    arb_enabled: bool = False
    arb_min_sum_ask_cents: int = 97
    arb_min_net_edge_cents: int = 1
    arb_min_entry_cents: int = 20
    arb_max_entry_cents: int = 80
    arb_max_quantity: int = 3
    arb_cooldown_s: float = 5.0
    # Sleeve dispatch policy:
    # - "parallel": evaluate all sleeves and publish all signals
    # - "flow_first": scalper/arb entry signals take priority; hold entries are fallback
    sleeve_mode: str = "flow_first"

    # ── Paper trading realism ────────────────────────────────────────────
    # Cents added/subtracted to paper fill prices to simulate the slippage
    # left after delayed paper execution re-prices against the future book.
    # Buys fill this many cents worse (higher price); sells receive this
    # many cents less after the post-latency book is evaluated.
    paper_slippage_cents: int = 1
    # Simulated delay between signal creation and paper order reaching the book.
    # 0/0 preserves legacy immediate-fill behavior for tests.
    paper_order_latency_min_ms: int = 0
    paper_order_latency_max_ms: int = 0
    # When True, deduct estimated taker fees from paper PnL using Kalshi's
    # fee schedule. Strongly recommended for realistic scalp evaluation.
    paper_apply_fees: bool = False

    # ── Execution tuning ───────────────────────────────────────────────
    order_timeout_ms: int = 300
    # "aggressive" (cross spread, current default) or "passive" (post at
    # best bid, longer timeout).  May also be set per-signal in future.
    default_order_style: str = "aggressive"
    # Timeout for passive orders (ms) — longer than aggressive since we
    # expect to queue rather than fill immediately.
    passive_order_timeout_ms: int = 5000

    # ── Truth feed ─────────────────────────────────────────────────────
    use_proxy_truth_feed: bool = False
    use_coinbase_ws: bool = True
    use_luzia_fallback: bool = True
    truth_feed_topic: str = "btc.mid_price"
    truth_feed_stale_timeout_s: float = 30.0  # halt if no tick for this long
    # Use fallback feed when primary (Coinbase) has been silent this long. Keeps switch
    # to OKX/Luzia instant (single timestamp check). Must be <= truth_feed_stale_timeout_s.
    fallback_activation_s: float = 5.0
    # Multi-asset feed config. If empty, runner falls back to legacy
    # truth_feed_topic BTC-only behavior.
    truth_feeds: List[TruthFeedConfig] = field(default_factory=list)
    assets: List[str] = field(default_factory=lambda: ["BTC"])
    window_minutes: List[int] = field(default_factory=lambda: [15, 60])
    include_range_markets: bool = False

    # ── Performance tuning ─────────────────────────────────────────────
    # Max tickers to subscribe on WebSocket (near-money filtering).
    # Limits task count and bus traffic to prevent event loop starvation.
    max_ws_markets: int = 120
    # Markets outside this fraction of current price are skipped in
    # probability recomputation (e.g. 0.08 = 8%).
    near_money_pct: float = 0.08

    # ── Settlement tracking ───────────────────────────────────────────
    enable_settlement_tracker: bool = True
    decision_tape_enabled: bool = False
    decision_tape_path: str = "logs/decision_tape/tape_{run_id}.jsonl"
    decision_tape_signal_sample_rate: float = 1.0
    decision_tape_rejection_sample_rate: float = 0.10

    # ── Luzia Fallback (legacy) ───────────────────────────────────────
    luzia_api_key: str = ""
    luzia_endpoints: List[str] = field(
        # Format: exchange/symbol
        default_factory=lambda: ["binance/BTC-USDT", "coinbase/BTC-USDT"]
    )

    # ── OKX WebSocket Fallback ────────────────────────────────────────
    # When True, use OKX public WebSocket for BTC price fallback instead of Luzia.
    use_okx_fallback: bool = False
    # OKX instrument IDs for ticker subscription (public channel, no API key required).
    okx_ticker_inst_ids: List[str] = field(default_factory=lambda: ["BTC-USDT"])
    # Optional: override WebSocket URL (default: wss://ws.okx.com:8443/ws/v5/public).
    okx_ws_url: str = ""

    # ── NTP / clock drift ──────────────────────────────────────────────
    enable_clock_offset_calibration: bool = True
    # Maximum acceptable offset (ms).  If calibration returns a larger
    # absolute value we log a warning and fall back to zero — the offset
    # is likely noise from a CDN proxy rather than real clock drift.
    max_clock_offset_ms: int = 5000

    # ── WebSocket feature flag ─────────────────────────────────────────
    # When False the system will NOT trade via WS-fed signals; it will
    # still connect (for orderbook / fills) but execution is disabled.
    # Use this until WS auth is confirmed against production.
    ws_trading_enabled: bool = False

    # ── WS auth signing path override ──────────────────────────────────
    # The path signed during the WS handshake.  Kalshi's REST signing
    # uses the request path (e.g. "/trade-api/v2/markets").  For WS the
    # signed path MAY differ (e.g. "/trade-api/ws/v2").  This field
    # lets you toggle without code changes once prod behaviour is
    # confirmed.  Set to "" to derive from base_url_ws automatically.
    ws_signing_path: str = ""

    # ── Dry-run mode ───────────────────────────────────────────────────
    # When True, signals are computed and logged but no orders are sent.
    dry_run: bool = True

    # ── Shutdown behaviour ───────────────────────────────────────────
    cancel_on_shutdown: bool = True

    # ── Visualization ──────────────────────────────────────────────
    visualizer_enabled: bool = True
    # "inline" = UI runs in same process (can starve event loop).
    # "separate" = UI runs in another process; trading process runs IPC server (default).
    visualizer_process: str = "separate"
    # IPC server bind when visualizer_process == "separate".
    ipc_bind: str = "127.0.0.1"
    ipc_port: int = 9999
    # When True, separate-process mode launches the terminal UI automatically in
    # a second console window connected to the IPC server.
    auto_launch_terminal_ui: bool = False
    # When False, separate-process mode runs IPC only and does not start the browser dashboard.
    dashboard_enabled: bool = True
    # Web dashboard port (decoupled UI in browser); only when visualizer_process == "separate"
    # and dashboard_enabled is True.
    dashboard_port: int = 9998
    # Re-measure Kalshi REST RTT every this many seconds for the terminal UI.
    kalshi_rtt_poll_interval_s: float = 5.0

    # ── Population management ──────────────────────────────────────────
    # When True, enables epoch-based retire/reseed and drawdown retire.
    enable_population_manager: bool = False
    # Fraction of reseeded bots that are perturbations of top performers
    # vs random exploration.  0.80 = 80% exploit, 20% explore.
    population_exploit_fraction: float = 0.80
    # Duration of one evaluation epoch in minutes.
    population_epoch_minutes: float = 60.0
    # Fraction of bots retired at each epoch end (bottom N% by robustness).
    population_retire_bottom_pct: float = 0.20
    # Hard-stop drawdown threshold per bot (fraction of start equity).
    # e.g. 0.15 = retire when drawdown exceeds 15% of $5,000 = $750.
    drawdown_retire_pct: float = 0.15
    # Scenario profile for this bot's execution assumptions.
    # "best" | "base" | "stress"  — see simulation.py for definitions.
    scenario_profile: str = "base"
    # Families allowed to emit live trade signals.
    # Empty = allow all families.
    # Example values: "BTC 15m", "BTC 60m", "BTC Range", "ETH 15m", "ETH 60m".
    live_families: List[str] = field(default_factory=list)
    # Families that run in shadow mode (signals evaluated but not executed).
    # Useful for diagnostics while isolating live PnL to stronger families.
    shadow_families: List[str] = field(default_factory=list)
    # Asset-level coarse allow-list. Empty = allow all discovered assets.
    trade_enabled_assets: List[str] = field(default_factory=list)
    # Asset-level shadow mode (evaluated but not executed).
    shadow_assets: List[str] = field(default_factory=list)
    # Evolution safety: require at least this many settled trades before a bot
    # is eligible for epoch ranking/retirement.
    population_min_trades_for_eval: int = 20
    # Evolution safety: require at least this many settled trades before
    # drawdown retire/reseed checks can trigger.
    population_min_trades_for_drawdown: int = 8

    # ── CPU compute threads ─────────────────────────────────────────────
    # Number of worker threads for CPU-bound work (e.g. probability recompute).
    # 0 = auto: min(32, cpu_count * 2). Set to a positive int to override.
    compute_threads: int = 0

    # ── Regime-aware gating ─────────────────────────────────────────────
    # Master toggle.  When False, all regime gates pass (backward compatible).
    enable_regime_gating: bool = False
    # Scalp gating under VOL_SPIKE: stricter microstructure requirements.
    scalp_spike_min_edge_cents: int = 8
    scalp_spike_max_spread_cents: int = 1
    scalp_spike_depth_min: int = 100
    scalp_spike_reprice_min: int = 6
    # Size reduction multiplier for scalps during VOL_SPIKE.
    scalp_spike_qty_multiplier: float = 0.5
    # Shorter max hold for spike scalps (minutes).
    scalp_spike_max_hold_minutes: float = 5.0
    # Hold gating under VOL_SPIKE: minimum edge for spike hold entries.
    hold_spike_min_edge: float = 0.06
    # Maximum seconds-to-settle for spike hold entries (block early entries).
    hold_spike_entry_horizon_s: float = 300.0
    # Fallback mode when regime data is unavailable.
    # "conservative" = treat as VOL_SPIKE + LIQ_LOW (blocks most entries).
    # "permissive" = allow all (effectively disables gating).
    regime_fallback_mode: str = "conservative"
    # Size cap multiplier under RISK_OFF.
    risk_off_qty_multiplier: float = 0.5

    # ── Per-family adaptive allocation ──────────────────────────────────
    # Master toggle: when True, enables per-family weight management,
    # per-family epoch ranking, and adaptive allocation.
    # Phase 2 kill switch — disable to fall back to global ranking.
    family_population_enabled: bool = False
    # Minimum weight any single family can hold (fraction, e.g. 0.05 = 5%).
    # Guarantees every family always gets continued exploration.
    family_min_weight: float = 0.05
    # Maximum weight any single family can hold.
    # Prevents over-concentration in one market type.
    family_max_weight: float = 0.40
    # How often (minutes) to recompute family weights from recent EMA performance.
    family_rebalance_interval_minutes: float = 30.0
    # Fraction of per-family reseeded bots that use exploration (random params).
    family_explore_fraction: float = 0.20
    # Fraction of per-family reseeded bots that use exploitation (perturbed survivors).
    family_exploit_fraction: float = 0.80
    # Phase 3 kill switch: when True, enables cross-asset context features
    # for confidence/size adjustment. Context never forces trade side.
    family_context_features_enabled: bool = False
    # Side-level protection in farm mode.
    # Temporarily blocks a (ticker, side) after persistent poor outcomes.
    side_guard_enabled: bool = False
    side_guard_min_settles: int = 80
    side_guard_max_win_rate: float = 0.15
    side_guard_min_avg_pnl_usd: float = -1.0
    side_guard_block_minutes: float = 30.0
    # Market-side entry caps (Phase 1)
    # Applies to each (market_ticker, side) during evaluator fan-out.
    enable_market_side_caps: bool = False
    market_side_cap_contracts: int = 0
    market_side_cap_usd: float = 0.0
    family_side_cap_usd: float = 0.0
    market_side_cap_enforcement_mode: str = "block"  # block | scale
    # Parameter-region penalties (Phase 2)
    enable_param_region_penalties: bool = False
    param_region_window_settles: int = 300
    param_region_min_samples: int = 80
    # Region is considered persistently losing when rolling SUM(net_pnl_usd)
    # over the configured window is <= this threshold.
    param_region_loss_threshold_usd: float = -100.0
    # Downweight multiplier for losing regions in downweight mode.
    param_region_penalty_factor: float = 0.5
    # Soft boost multiplier for persistently winning regions in downweight mode.
    param_region_gain_factor: float = 1.10
    # Minimum samples required before applying context-level gain/loss sizing.
    param_region_context_min_samples: int = 300
    # Context-level average PnL thresholds (USD/trade) for soft sizing.
    param_region_loss_avg_pnl_usd: float = -2.0
    param_region_gain_avg_pnl_usd: float = 2.0
    # Cooldown block duration for losing regions in cooldown_block mode.
    param_region_block_minutes: float = 30.0
    param_region_mode: str = "downweight"  # downweight | cooldown_block
    # Retuned region knobs (Phase 2B): target low-edge / low-mid entry regimes.
    param_region_low_edge_max: float = 0.07
    param_region_low_entry_max_cents: int = 55
    param_region_low_entry_floor_cents: int = 40
    param_region_enable_family_side: bool = True
    # Family allow-list for high max_entry_cents regions (>=78).
    # Empty means no exceptions.
    param_region_allow_high_entry_families: List[str] = field(default_factory=list)
    # Crowding throttle (Phase 4; can run diagnostics-only while disabled)
    enable_crowding_throttle: bool = False
    crowding_window_s: float = 10.0
    crowding_fills_per_sec_threshold: float = 4.0
    crowding_qty_multiplier: float = 0.5
    crowding_pause_s: float = 5.0
    crowding_mode: str = "scale"  # scale | pause
    # Candidate-region weighting (Phase 3)
    # Bias farm parameter sampling toward this region while preserving exploration.
    candidate_region_enabled: bool = False
    candidate_region_weight: float = 0.8
    candidate_region_explore_floor: float = 0.2
    candidate_region_min_edge_min: float = 0.07
    candidate_region_min_edge_max: float = 0.09
    candidate_region_persistence_min_ms: int = 120
    candidate_region_persistence_max_ms: int = 300
    candidate_region_entry_min_cents_min: int = 40
    candidate_region_entry_min_cents_max: int = 45
    candidate_region_entry_max_cents_min: int = 70
    candidate_region_entry_max_cents_max: int = 70

    # ── Context policy engine (soft allocation) ──────────────────────────
    # Master toggle for the weighted context policy engine.
    # Replaces hard family drops with core/explore lane allocation.
    enable_context_policy: bool = False
    # Rolling settlement window size for context weight updates.
    context_policy_window_settles: int = 500
    # Minimum samples before a context key gets non-default weight.
    context_policy_min_samples: int = 50
    # Shrinkage toward 1.0 for low-sample contexts (0=no shrinkage, 1=full).
    context_policy_shrinkage: float = 0.7
    # Weight bounds: core lane (promoted) and explore lane (probation).
    context_policy_core_weight_min: float = 1.0
    context_policy_core_weight_max: float = 1.5
    context_policy_challenger_weight: float = 0.9
    context_policy_explore_weight: float = 0.5
    # Expectancy threshold (USD/trade) for core promotion.
    context_policy_promote_threshold_usd: float = 0.50
    # Expectancy threshold (USD/trade) for demotion to explore.
    context_policy_demote_threshold_usd: float = -0.50
    # Hard cap on unique context keys tracked in-memory (bounds cardinality).
    context_policy_max_keys: int = 10000
    # Path to persisted policy file (JSON). Empty = no persistence.
    context_policy_file: str = "config/kalshi_context_policy.json"
    # Auto-reload policy file while running to reduce policy staleness.
    context_policy_auto_reload: bool = False
    context_policy_reload_interval_s: float = 300.0
    # Optional lane-share controller (Core/Challenger/Explore target mix).
    context_policy_share_control_enabled: bool = False
    context_policy_core_target_share: float = 0.65
    context_policy_challenger_target_share: float = 0.25
    context_policy_explore_target_share: float = 0.10
    context_policy_share_window_calls: int = 20000
    context_policy_share_control_gain: float = 0.5
    context_policy_share_control_min_mult: float = 0.6
    context_policy_share_control_max_mult: float = 1.4

    # ── Strike distance context features ─────────────────────────────────
    # Bucket edges for strike_distance_pct (fraction of spot price).
    strike_distance_bucket_edges: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.02, 0.05]
    )
    # near_money penalty mode: "off" | "soft" | "hard"
    # "soft" applies a multiplier; "hard" blocks; "off" = no effect.
    near_money_penalty_mode: str = "off"
    # Multiplier applied when near_money=True and mode="soft".
    near_money_penalty_multiplier: float = 0.8

    # ── Context drift guard ──────────────────────────────────────────────
    # Detect promoted contexts whose expectancy drifts negative.
    enable_drift_guard: bool = False
    # Comparison window size in settlements.
    drift_guard_window_settles: int = 200
    # Hard cap on unique context keys tracked in-memory.
    drift_guard_max_keys: int = 10000
    # Consecutive negative windows before auto-demotion.
    drift_guard_consecutive_negative: int = 3
    # Threshold: expectancy must be below this to count as negative.
    drift_guard_negative_threshold_usd: float = -0.25
    # Weight multiplier reduction on demotion.
    drift_guard_demote_multiplier: float = 0.5

    # ── Adaptive concentration hardening ─────────────────────────────────
    # Tighten caps when concentration + negative expectancy persist.
    enable_adaptive_caps: bool = False
    # Minimum settlements before adaptive cap check.
    adaptive_cap_min_samples: int = 100
    # Hard cap on unique context keys tracked in-memory.
    adaptive_cap_max_keys: int = 10000
    # Negative expectancy threshold to trigger cap tightening.
    adaptive_cap_negative_threshold_usd: float = -0.50
    # Cap tightening multiplier (applied to market_side_cap).
    adaptive_cap_tightening_mult: float = 0.5
    # Key cooldown duration (minutes) when tightened.
    adaptive_cap_cooldown_minutes: float = 30.0

    # ── Execution quality / edge retention ───────────────────────────────
    # Track expected vs realized edge per context.
    enable_edge_tracking: bool = False
    # Rolling window for edge retention computation.
    edge_tracking_window_settles: int = 300
    # Hard cap on unique context keys tracked in-memory.
    edge_tracking_max_keys: int = 10000
    # Minimum samples before edge retention affects weights.
    edge_tracking_min_samples: int = 50
    # If realized/expected ratio is below this, downweight.
    edge_retention_decay_threshold: float = 0.3
    # Downweight multiplier for poor edge retention.
    edge_retention_decay_multiplier: float = 0.7

    # ── Bot population scaling ───────────────────────────────────────────
    # Staged scaling gates — do NOT blindly increase bot count.
    bot_population_scale_enabled: bool = False
    # Scale stages as multipliers of base bot count.
    bot_population_scale_schedule: List[float] = field(
        default_factory=lambda: [1.0, 1.15, 1.30, 1.50]
    )
    # Minimum hours at current stage before scale-up attempt.
    bot_population_scale_min_window_hours: float = 8.0
    # Max multiplier step per stage transition.
    bot_population_scale_max_step: float = 0.20
    # Consecutive passing windows required before scale-up.
    bot_population_scale_require_passes: int = 2
    # Concentration gate: top market-side share must be below this.
    bot_population_scale_max_concentration: float = 0.30
    # Edge retention gate: must be above this ratio.
    bot_population_scale_min_edge_retention: float = 0.3
    # Performance gate: minimum expectancy (USD/trade).
    bot_population_scale_min_expectancy: float = 0.0
    # Drawdown gate: max drawdown fraction allowed.
    bot_population_scale_max_drawdown: float = 0.10
    # Cooldown hours after scale-down before re-attempting scale-up.
    bot_population_scale_cooldown_hours: float = 16.0

    def __post_init__(self) -> None:
        if not self.bot_id.strip():
            raise ValueError("bot_id must be non-empty")
        if self.bankroll_usd <= 0:
            raise ValueError("bankroll_usd must be positive")
        if not (0 < self.max_fraction_per_market <= 1):
            raise ValueError("max_fraction_per_market must be in (0, 1]")
        if not (0 < self.daily_drawdown_limit <= 1):
            raise ValueError("daily_drawdown_limit must be in (0, 1]")
        if self.min_edge_threshold < 0:
            raise ValueError("min_edge_threshold must be non-negative")
        if not (0 < self.risk_fraction_per_trade <= 0.1):
            raise ValueError("risk_fraction_per_trade must be in (0, 0.1]")
        if not (0 < self.sizing_risk_fraction <= 0.1):
            raise ValueError("sizing_risk_fraction must be in (0, 0.1]")
        if self.consecutive_drawdown_limit < 1:
            raise ValueError("consecutive_drawdown_limit must be >= 1")
        if not (0 <= self.effective_edge_fee_pct <= 0.5):
            raise ValueError("effective_edge_fee_pct must be in [0, 0.5]")
        if self.persistence_window_ms < 0:
            raise ValueError("persistence_window_ms must be non-negative")
        if self.latency_circuit_breaker_ms < 0:
            raise ValueError("latency_circuit_breaker_ms must be non-negative")
        if self.fallback_activation_s <= 0 or self.fallback_activation_s > self.truth_feed_stale_timeout_s:
            raise ValueError("fallback_activation_s must be in (0, truth_feed_stale_timeout_s]")
        if self.signal_cooldown_s < 0:
            raise ValueError("signal_cooldown_s must be non-negative")
        if self.max_contracts_per_ticker < 0:
            raise ValueError("max_contracts_per_ticker must be non-negative")
        if self.base_contract_qty < 1:
            raise ValueError("base_contract_qty must be >= 1")
        if not (0 <= self.min_entry_cents <= 100):
            raise ValueError("min_entry_cents must be in [0, 100]")
        if not (0 <= self.max_entry_cents <= 100):
            raise ValueError("max_entry_cents must be in [0, 100]")
        if self.min_entry_cents > self.max_entry_cents:
            raise ValueError("min_entry_cents must be <= max_entry_cents")
        if not (0 <= self.no_avoid_above_cents <= 100):
            raise ValueError("no_avoid_above_cents must be in [0, 100]")
        if not (0 <= self.hold_tail_penalty_start_cents <= 100):
            raise ValueError("hold_tail_penalty_start_cents must be in [0, 100]")
        if not (0.0 <= self.hold_tail_penalty_per_10c <= 0.5):
            raise ValueError("hold_tail_penalty_per_10c must be in [0, 0.5]")
        if self.hold_min_net_edge_cents < 0:
            raise ValueError("hold_min_net_edge_cents must be non-negative")
        if self.hold_entry_cost_buffer_cents < 0:
            raise ValueError("hold_entry_cost_buffer_cents must be non-negative")
        if not (0.0 <= self.hold_min_divergence_threshold <= 1.0):
            raise ValueError("hold_min_divergence_threshold must be in [0, 1]")
        if not (0.0 <= self.hold_flow_reversal_threshold <= 1.0):
            raise ValueError("hold_flow_reversal_threshold must be in [0, 1]")
        if not (0.0 <= self.obi_p_yes_bias_weight <= 1.0):
            raise ValueError("obi_p_yes_bias_weight must be in [0, 1]")
        if not (0.0 <= self.momentum_p_yes_bias_weight <= 2000.0):
            raise ValueError("momentum_p_yes_bias_weight must be in [0, 2000]")
        if not (0.0 <= self.trade_flow_p_yes_bias_weight <= 1.0):
            raise ValueError("trade_flow_p_yes_bias_weight must be in [0, 1]")
        if self.yes_avoid_min_cents < 0 or self.yes_avoid_max_cents < 0:
            raise ValueError("yes_avoid_min/max_cents must be non-negative")
        if self.yes_avoid_min_cents > self.yes_avoid_max_cents and self.yes_avoid_max_cents != 0:
            raise ValueError("yes_avoid_min_cents must be <= yes_avoid_max_cents")
        if self.max_entry_minutes_to_expiry < 0:
            raise ValueError("max_entry_minutes_to_expiry must be non-negative")
        if self.hold_min_entry_minutes_to_expiry < 0:
            raise ValueError("hold_min_entry_minutes_to_expiry must be non-negative")
        if self.range_max_entry_minutes_to_expiry < 0:
            raise ValueError("range_max_entry_minutes_to_expiry must be non-negative")
        if self.scalp_min_profit_cents < 1:
            raise ValueError("scalp_min_profit_cents must be >= 1")
        if self.scalp_stop_loss_cents < 0:
            raise ValueError("scalp_stop_loss_cents must be non-negative")
        if self.scalp_max_hold_minutes <= 0:
            raise ValueError("scalp_max_hold_minutes must be positive")
        if not (0 <= self.scalp_min_entry_cents <= 100):
            raise ValueError("scalp_min_entry_cents must be in [0, 100]")
        if not (0 <= self.scalp_max_entry_cents <= 100):
            raise ValueError("scalp_max_entry_cents must be in [0, 100]")
        if self.scalp_min_entry_cents > self.scalp_max_entry_cents:
            raise ValueError("scalp_min_entry_cents must be <= scalp_max_entry_cents")
        if self.scalp_max_quantity < 1:
            raise ValueError("scalp_max_quantity must be >= 1")
        if self.scalp_min_edge_cents < 1:
            raise ValueError("scalp_min_edge_cents must be >= 1")
        if self.scalp_max_spread_cents < 0:
            raise ValueError("scalp_max_spread_cents must be non-negative")
        if self.scalp_cooldown_s < 0:
            raise ValueError("scalp_cooldown_s must be non-negative")
        if self.scalp_momentum_min_drift < 0:
            raise ValueError("scalp_momentum_min_drift must be non-negative")
        if not (0.0 <= self.scalp_directional_score_threshold <= 1.0):
            raise ValueError("scalp_directional_score_threshold must be in [0, 1]")
        if self.scalp_directional_drift_scale <= 0:
            raise ValueError("scalp_directional_drift_scale must be positive")
        if any(
            w < 0.0
            for w in (
                self.scalp_directional_drift_weight,
                self.scalp_directional_obi_weight,
                self.scalp_directional_flow_weight,
                self.scalp_directional_depth_weight,
                self.scalp_directional_delta_yes_weight,
                self.scalp_directional_delta_no_weight,
            )
        ):
            raise ValueError("scalp_directional_*_weight must be non-negative")
        if self.momentum_min_reprice_move_cents < 1:
            raise ValueError("momentum_min_reprice_move_cents must be >= 1")
        if self.momentum_reprice_window_s <= 0:
            raise ValueError("momentum_reprice_window_s must be positive")
        if not (0.0 <= self.momentum_min_orderbook_imbalance <= 1.0):
            raise ValueError("momentum_min_orderbook_imbalance must be in [0, 1]")
        if self.momentum_max_spread_cents < 0:
            raise ValueError("momentum_max_spread_cents must be non-negative")
        if self.momentum_min_edge_cents < -20:
            raise ValueError("momentum_min_edge_cents must be >= -20")
        if self.momentum_min_profit_cents < 1:
            raise ValueError("momentum_min_profit_cents must be >= 1")
        if self.momentum_entry_cost_buffer_cents < 0:
            raise ValueError("momentum_entry_cost_buffer_cents must be non-negative")
        if self.momentum_max_quantity < 1:
            raise ValueError("momentum_max_quantity must be >= 1")
        if not (0 <= self.arb_min_sum_ask_cents <= 200):
            raise ValueError("arb_min_sum_ask_cents must be in [0, 200]")
        if self.arb_min_net_edge_cents < 0:
            raise ValueError("arb_min_net_edge_cents must be non-negative")
        if not (0 <= self.arb_min_entry_cents <= 100):
            raise ValueError("arb_min_entry_cents must be in [0, 100]")
        if not (0 <= self.arb_max_entry_cents <= 100):
            raise ValueError("arb_max_entry_cents must be in [0, 100]")
        if self.arb_min_entry_cents > self.arb_max_entry_cents:
            raise ValueError("arb_min_entry_cents must be <= arb_max_entry_cents")
        if self.arb_max_quantity < 1:
            raise ValueError("arb_max_quantity must be >= 1")
        if self.arb_cooldown_s < 0:
            raise ValueError("arb_cooldown_s must be non-negative")
        if self.sleeve_mode not in {"parallel", "flow_first"}:
            raise ValueError("sleeve_mode must be one of: parallel, flow_first")
        if self.paper_slippage_cents < 0:
            raise ValueError("paper_slippage_cents must be non-negative")
        if not (0.0 <= self.decision_tape_signal_sample_rate <= 1.0):
            raise ValueError("decision_tape_signal_sample_rate must be in [0, 1]")
        if not (0.0 <= self.decision_tape_rejection_sample_rate <= 1.0):
            raise ValueError("decision_tape_rejection_sample_rate must be in [0, 1]")
        if not self.decision_tape_path:
            raise ValueError("decision_tape_path must be non-empty")
        if self.paper_order_latency_min_ms < 0 or self.paper_order_latency_max_ms < 0:
            raise ValueError("paper_order_latency_min/max_ms must be non-negative")
        if self.paper_order_latency_min_ms > self.paper_order_latency_max_ms:
            raise ValueError("paper_order_latency_min_ms must be <= paper_order_latency_max_ms")
        if self.kalshi_rtt_poll_interval_s < 0:
            raise ValueError("kalshi_rtt_poll_interval_s must be non-negative")
        if self.rate_limit_read_per_sec <= 0 or self.rate_limit_write_per_sec <= 0:
            raise ValueError("rate limits must be positive")
        if not (0.0 <= self.population_exploit_fraction <= 1.0):
            raise ValueError("population_exploit_fraction must be in [0, 1]")
        if self.population_epoch_minutes <= 0:
            raise ValueError("population_epoch_minutes must be positive")
        if not (0.0 < self.population_retire_bottom_pct <= 1.0):
            raise ValueError("population_retire_bottom_pct must be in (0, 1]")
        if not (0.0 < self.drawdown_retire_pct <= 1.0):
            raise ValueError("drawdown_retire_pct must be in (0, 1]")
        if self.scenario_profile not in ("best", "base", "stress"):
            raise ValueError("scenario_profile must be 'best', 'base', or 'stress'")
        if self.population_min_trades_for_eval < 0:
            raise ValueError("population_min_trades_for_eval must be non-negative")
        if self.population_min_trades_for_drawdown < 0:
            raise ValueError("population_min_trades_for_drawdown must be non-negative")
        supported_assets = {"BTC", "ETH", "SOL"}
        if any(a.upper() not in supported_assets for a in self.assets):
            raise ValueError("assets must be a subset of BTC/ETH/SOL")
        if any(a.upper() not in supported_assets for a in self.trade_enabled_assets):
            raise ValueError("trade_enabled_assets must be a subset of BTC/ETH/SOL")
        if any(a.upper() not in supported_assets for a in self.shadow_assets):
            raise ValueError("shadow_assets must be a subset of BTC/ETH/SOL")
        if any(m not in (15, 60) for m in self.window_minutes):
            raise ValueError("window_minutes entries must be 15 or 60")
        # Regime gating validation
        if self.scalp_spike_min_edge_cents < 1:
            raise ValueError("scalp_spike_min_edge_cents must be >= 1")
        if self.scalp_spike_max_spread_cents < 0:
            raise ValueError("scalp_spike_max_spread_cents must be non-negative")
        if self.scalp_spike_depth_min < 0:
            raise ValueError("scalp_spike_depth_min must be non-negative")
        if self.scalp_spike_reprice_min < 0:
            raise ValueError("scalp_spike_reprice_min must be non-negative")
        if not (0.0 < self.scalp_spike_qty_multiplier <= 1.0):
            raise ValueError("scalp_spike_qty_multiplier must be in (0, 1]")
        if self.scalp_spike_max_hold_minutes <= 0:
            raise ValueError("scalp_spike_max_hold_minutes must be positive")
        if self.hold_spike_min_edge < 0:
            raise ValueError("hold_spike_min_edge must be non-negative")
        if self.hold_spike_entry_horizon_s <= 0:
            raise ValueError("hold_spike_entry_horizon_s must be positive")
        if self.regime_fallback_mode not in ("conservative", "permissive"):
            raise ValueError("regime_fallback_mode must be 'conservative' or 'permissive'")
        if not (0.0 < self.risk_off_qty_multiplier <= 1.0):
            raise ValueError("risk_off_qty_multiplier must be in (0, 1]")
        # Family adaptive allocation validation
        if not (0.0 < self.family_min_weight <= 0.5):
            raise ValueError("family_min_weight must be in (0, 0.5]")
        if not (self.family_min_weight <= self.family_max_weight <= 1.0):
            raise ValueError("family_max_weight must be in [family_min_weight, 1.0]")
        if self.family_rebalance_interval_minutes <= 0:
            raise ValueError("family_rebalance_interval_minutes must be positive")
        if not (0.0 <= self.family_explore_fraction <= 1.0):
            raise ValueError("family_explore_fraction must be in [0, 1]")
        if not (0.0 <= self.family_exploit_fraction <= 1.0):
            raise ValueError("family_exploit_fraction must be in [0, 1]")
        if self.side_guard_min_settles < 10:
            raise ValueError("side_guard_min_settles must be >= 10")
        if not (0.0 <= self.side_guard_max_win_rate <= 1.0):
            raise ValueError("side_guard_max_win_rate must be in [0, 1]")
        if self.side_guard_block_minutes <= 0:
            raise ValueError("side_guard_block_minutes must be positive")
        if self.market_side_cap_contracts < 0:
            raise ValueError("market_side_cap_contracts must be >= 0")
        if self.market_side_cap_usd < 0:
            raise ValueError("market_side_cap_usd must be >= 0")
        if self.family_side_cap_usd < 0:
            raise ValueError("family_side_cap_usd must be >= 0")
        if self.market_side_cap_enforcement_mode not in ("block", "scale"):
            raise ValueError("market_side_cap_enforcement_mode must be 'block' or 'scale'")
        if self.param_region_window_settles <= 0:
            raise ValueError("param_region_window_settles must be positive")
        if self.param_region_min_samples <= 0:
            raise ValueError("param_region_min_samples must be positive")
        if not (0.0 < self.param_region_penalty_factor <= 1.0):
            raise ValueError("param_region_penalty_factor must be in (0, 1]")
        if not (1.0 <= self.param_region_gain_factor <= 2.0):
            raise ValueError("param_region_gain_factor must be in [1, 2]")
        if self.param_region_context_min_samples <= 0:
            raise ValueError("param_region_context_min_samples must be positive")
        if self.param_region_block_minutes <= 0:
            raise ValueError("param_region_block_minutes must be positive")
        if self.param_region_mode not in ("downweight", "cooldown_block"):
            raise ValueError("param_region_mode must be 'downweight' or 'cooldown_block'")
        if not (0.0 <= self.param_region_low_edge_max <= 1.0):
            raise ValueError("param_region_low_edge_max must be in [0, 1]")
        if not (0 <= self.param_region_low_entry_floor_cents <= 99):
            raise ValueError("param_region_low_entry_floor_cents must be in [0, 99]")
        if not (0 <= self.param_region_low_entry_max_cents <= 99):
            raise ValueError("param_region_low_entry_max_cents must be in [0, 99]")
        if self.param_region_low_entry_floor_cents > self.param_region_low_entry_max_cents:
            raise ValueError("param_region_low_entry_floor_cents must be <= param_region_low_entry_max_cents")
        if self.crowding_window_s <= 0:
            raise ValueError("crowding_window_s must be positive")
        if self.crowding_fills_per_sec_threshold <= 0:
            raise ValueError("crowding_fills_per_sec_threshold must be positive")
        if not (0.0 < self.crowding_qty_multiplier <= 1.0):
            raise ValueError("crowding_qty_multiplier must be in (0, 1]")
        if self.crowding_pause_s < 0:
            raise ValueError("crowding_pause_s must be non-negative")
        if self.crowding_mode not in ("scale", "pause"):
            raise ValueError("crowding_mode must be 'scale' or 'pause'")
        if not (0.0 <= self.candidate_region_weight <= 1.0):
            raise ValueError("candidate_region_weight must be in [0, 1]")
        if not (0.0 <= self.candidate_region_explore_floor <= 1.0):
            raise ValueError("candidate_region_explore_floor must be in [0, 1]")
        if not (0.0 <= self.candidate_region_min_edge_min <= 1.0):
            raise ValueError("candidate_region_min_edge_min must be in [0, 1]")
        if not (0.0 <= self.candidate_region_min_edge_max <= 1.0):
            raise ValueError("candidate_region_min_edge_max must be in [0, 1]")
        if self.candidate_region_min_edge_min > self.candidate_region_min_edge_max:
            raise ValueError("candidate_region_min_edge_min must be <= candidate_region_min_edge_max")
        if self.candidate_region_persistence_min_ms < 0 or self.candidate_region_persistence_max_ms < 0:
            raise ValueError("candidate_region_persistence_min_ms/max_ms must be non-negative")
        if self.candidate_region_persistence_min_ms > self.candidate_region_persistence_max_ms:
            raise ValueError(
                "candidate_region_persistence_min_ms must be <= candidate_region_persistence_max_ms"
            )
        if not (0 <= self.candidate_region_entry_min_cents_min <= 100):
            raise ValueError("candidate_region_entry_min_cents_min must be in [0, 100]")
        if not (0 <= self.candidate_region_entry_min_cents_max <= 100):
            raise ValueError("candidate_region_entry_min_cents_max must be in [0, 100]")
        if self.candidate_region_entry_min_cents_min > self.candidate_region_entry_min_cents_max:
            raise ValueError(
                "candidate_region_entry_min_cents_min must be <= candidate_region_entry_min_cents_max"
            )
        if not (0 <= self.candidate_region_entry_max_cents_min <= 100):
            raise ValueError("candidate_region_entry_max_cents_min must be in [0, 100]")
        if not (0 <= self.candidate_region_entry_max_cents_max <= 100):
            raise ValueError("candidate_region_entry_max_cents_max must be in [0, 100]")
        if self.candidate_region_entry_max_cents_min > self.candidate_region_entry_max_cents_max:
            raise ValueError(
                "candidate_region_entry_max_cents_min must be <= candidate_region_entry_max_cents_max"
            )
        # Context policy validation
        if self.context_policy_window_settles <= 0:
            raise ValueError("context_policy_window_settles must be positive")
        if self.context_policy_min_samples <= 0:
            raise ValueError("context_policy_min_samples must be positive")
        if self.context_policy_max_keys <= 0:
            raise ValueError("context_policy_max_keys must be positive")
        if self.context_policy_reload_interval_s <= 0:
            raise ValueError("context_policy_reload_interval_s must be positive")
        if not (0.0 <= self.context_policy_shrinkage <= 1.0):
            raise ValueError("context_policy_shrinkage must be in [0, 1]")
        if self.context_policy_core_weight_min < 1.0:
            raise ValueError("context_policy_core_weight_min must be >= 1.0")
        if self.context_policy_core_weight_max < self.context_policy_core_weight_min:
            raise ValueError("context_policy_core_weight_max must be >= core_weight_min")
        if not (0.0 < self.context_policy_challenger_weight <= self.context_policy_core_weight_max):
            raise ValueError("context_policy_challenger_weight must be in (0, core_weight_max]")
        if not (0.0 < self.context_policy_explore_weight <= 1.0):
            raise ValueError("context_policy_explore_weight must be in (0, 1]")
        if self.context_policy_share_window_calls <= 0:
            raise ValueError("context_policy_share_window_calls must be positive")
        if not (0.0 <= self.context_policy_share_control_gain <= 2.0):
            raise ValueError("context_policy_share_control_gain must be in [0, 2]")
        if not (0.0 < self.context_policy_share_control_min_mult <= 1.0):
            raise ValueError("context_policy_share_control_min_mult must be in (0, 1]")
        if self.context_policy_share_control_max_mult < 1.0:
            raise ValueError("context_policy_share_control_max_mult must be >= 1.0")
        if self.context_policy_share_control_max_mult < self.context_policy_share_control_min_mult:
            raise ValueError("context_policy_share_control_max_mult must be >= min_mult")
        lane_target_sum = (
            self.context_policy_core_target_share
            + self.context_policy_challenger_target_share
            + self.context_policy_explore_target_share
        )
        if lane_target_sum <= 0:
            raise ValueError("context policy lane target shares must sum to > 0")
        if self.context_policy_core_target_share < 0:
            raise ValueError("context_policy_core_target_share must be >= 0")
        if self.context_policy_challenger_target_share < 0:
            raise ValueError("context_policy_challenger_target_share must be >= 0")
        if self.context_policy_explore_target_share < 0:
            raise ValueError("context_policy_explore_target_share must be >= 0")
        if self.near_money_penalty_mode not in ("off", "soft", "hard"):
            raise ValueError("near_money_penalty_mode must be 'off', 'soft', or 'hard'")
        if not (0.0 < self.near_money_penalty_multiplier <= 1.0):
            raise ValueError("near_money_penalty_multiplier must be in (0, 1]")
        # Drift guard validation
        if self.drift_guard_window_settles <= 0:
            raise ValueError("drift_guard_window_settles must be positive")
        if self.drift_guard_max_keys <= 0:
            raise ValueError("drift_guard_max_keys must be positive")
        if self.drift_guard_consecutive_negative < 1:
            raise ValueError("drift_guard_consecutive_negative must be >= 1")
        if not (0.0 < self.drift_guard_demote_multiplier <= 1.0):
            raise ValueError("drift_guard_demote_multiplier must be in (0, 1]")
        # Adaptive caps validation
        if self.adaptive_cap_min_samples <= 0:
            raise ValueError("adaptive_cap_min_samples must be positive")
        if self.adaptive_cap_max_keys <= 0:
            raise ValueError("adaptive_cap_max_keys must be positive")
        if not (0.0 < self.adaptive_cap_tightening_mult <= 1.0):
            raise ValueError("adaptive_cap_tightening_mult must be in (0, 1]")
        if self.adaptive_cap_cooldown_minutes <= 0:
            raise ValueError("adaptive_cap_cooldown_minutes must be positive")
        # Edge tracking validation
        if self.edge_tracking_window_settles <= 0:
            raise ValueError("edge_tracking_window_settles must be positive")
        if self.edge_tracking_min_samples <= 0:
            raise ValueError("edge_tracking_min_samples must be positive")
        if self.edge_tracking_max_keys <= 0:
            raise ValueError("edge_tracking_max_keys must be positive")
        if not (0.0 <= self.edge_retention_decay_threshold <= 1.0):
            raise ValueError("edge_retention_decay_threshold must be in [0, 1]")
        if not (0.0 < self.edge_retention_decay_multiplier <= 1.0):
            raise ValueError("edge_retention_decay_multiplier must be in (0, 1]")
        # Population scaling validation
        if self.bot_population_scale_min_window_hours < 0:
            raise ValueError("bot_population_scale_min_window_hours must be non-negative")
        if not (0.0 < self.bot_population_scale_max_step <= 1.0):
            raise ValueError("bot_population_scale_max_step must be in (0, 1]")
        if self.bot_population_scale_require_passes < 1:
            raise ValueError("bot_population_scale_require_passes must be >= 1")
        if self.bot_population_scale_cooldown_hours < 0:
            raise ValueError("bot_population_scale_cooldown_hours must be non-negative")


def load_config(raw: dict[str, Any]) -> KalshiConfig:
    # Build a *KalshiConfig* from an untyped dict.
    #
    # Unknown keys are silently ignored so callers can pass a superset of
    # configuration (e.g. an entire Argus config block).
    known = {f.name for f in KalshiConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known}
    if "risk_fraction_per_trade" not in filtered and "sizing_risk_fraction" in filtered:
        filtered["risk_fraction_per_trade"] = filtered["sizing_risk_fraction"]
    if "sizing_risk_fraction" not in filtered and "risk_fraction_per_trade" in filtered:
        filtered["sizing_risk_fraction"] = filtered["risk_fraction_per_trade"]
    if "market_side_cap_enforcement_mode" not in filtered and "cap_enforcement_mode" in filtered:
        filtered["market_side_cap_enforcement_mode"] = filtered["cap_enforcement_mode"]
    tf_raw = filtered.get("truth_feeds", [])
    if isinstance(tf_raw, list):
        parsed: List[TruthFeedConfig] = []
        for item in tf_raw:
            if isinstance(item, dict):
                parsed.append(TruthFeedConfig(
                    asset=str(item.get("asset", "BTC")).upper(),
                    topic=str(item.get("topic", "btc.mid_price")),
                    coinbase_symbol=str(item.get("coinbase_symbol", "BTC/USDT")),
                    publish_to_core_bus=bool(item.get("publish_to_core_bus", False)),
                ))
        filtered["truth_feeds"] = parsed
    for key in ("trade_enabled_assets", "shadow_assets"):
        vals = filtered.get(key)
        if isinstance(vals, list):
            filtered[key] = [str(v).upper() for v in vals if str(v).strip()]
    for key in ("live_families", "shadow_families"):
        vals = filtered.get(key)
        if isinstance(vals, list):
            filtered[key] = [str(v).strip() for v in vals if str(v).strip()]
    vals = filtered.get("param_region_allow_high_entry_families")
    if isinstance(vals, list):
        filtered["param_region_allow_high_entry_families"] = [
            str(v).strip() for v in vals if str(v).strip()
        ]
    # Coerce float-list config fields.
    for key in ("strike_distance_bucket_edges", "bot_population_scale_schedule"):
        vals = filtered.get(key)
        if isinstance(vals, list):
            filtered[key] = [float(v) for v in vals]
    return KalshiConfig(**filtered)
