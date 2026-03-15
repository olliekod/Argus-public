# Argus

Argus is a deterministic market-data, signal-generation, replay, and research platform with a separate Kalshi trading stack under `argus_kalshi/`. The core `src/` system focuses on ingesting market data, normalizing it into bars and option snapshots, computing regimes and features, generating signals, persisting everything to SQLite, and running offline evaluation. The `argus_kalshi/` package is a more specialized event-driven system for Kalshi BTC strike contracts, with its own probability model, execution engine, farm runner, and terminal UI.

The codebase is built around one idea: the same inputs should produce the same downstream state. Bars are built from exchange timestamps, replay packs slice historical state without lookahead, and the research loop reuses the same provider policy and schemas as the live collectors.

## Current Data Sources

Argus has connectors for more sources than it uses at once. The current policy in `config/config.yaml` makes these choices:

| Layer | Current primary source | Current secondary / fallback | Notes |
| --- | --- | --- | --- |
| 1-minute bars | Alpaca | Yahoo | `data_sources.bars_primary = alpaca`, `bars_secondary = yahoo` |
| Forward outcomes | Derived from bars_primary | None | Outcome engine computes labels from stored bars |
| Option chain snapshots | Public.com | Tastytrade | `options_snapshots_primary = public`, `options_snapshots_secondary = tastytrade` |
| Real-time option stream / Greeks | Tastytrade DXLink | None | Used for live IV and quote enrichment |
| Crypto truth feed for core monitoring | Coinbase WebSocket | OKX WebSocket fallback, Bybit auxiliary | Coinbase is the truth-critical path in the orchestrator |
| Crypto derivatives / auxiliary exchange data | Bybit, Deribit | OKX fallback | Bybit and Deribit are available in the core stack |
| Global macro daily bars | Alpha Vantage | None | Used for global risk flow |
| News sentiment | Yahoo Finance RSS, MarketWatch / Dow Jones RSS, optional NewsAPI key | None | Lexicon scorer plus feed aggregation |
| Prediction market watchlist research | Polymarket Gamma + CLOB | None | Optional and modular |
| Kalshi market data | Kalshi REST + WebSocket | Shared with local BTC truth feeds | Implemented in `argus_kalshi/` |

Important distinctions:

- Alpaca is the current primary source for 1-minute bars in the core Argus pipeline.
- Public.com is the current primary source for option snapshots and IV / Greeks snapshots.
- Tastytrade provides both secondary option snapshots and the DXLink streaming path for live option quotes and Greeks.
- Coinbase is treated as the primary truth feed for the Kalshi sidecar and truth-critical BTC price paths, with OKX as fallback.
- The repo contains connectors that are broader than the currently selected policy. The README below describes both the active defaults and the modules that are available.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | Core Argus platform: orchestration, connectors, detectors, analysis, persistence, replay, and dashboards |
| `argus_kalshi/` | Kalshi-specific trading stack, strategy engines, farm logic, UI, and IPC |
| `scripts/` | Operational CLI tools for backfills, verification, audits, research, and maintenance |
| `tests/` | Main pytest suite |
| `config/` | Runtime config files and secrets path expected by the loaders |
| `data/` | Runtime databases and static lexical resources |
| `logs/` | Runtime logs and reports |
| `docs/` | Notes, design docs, and command references |

## What the System Does

At a high level, the core Argus runtime does this:

1. Collect quotes, bars, and option chain data from configured providers.
2. Normalize those feeds into common events on a central event bus.
3. Aggregate quote ticks into deterministic 1-minute OHLCV bars.
4. Compute rolling returns, realized volatility, jump scores, trend, ATR, RSI, and liquidity proxies.
5. Classify symbol and market regimes.
6. Run detectors and strategy modules against those bars, regimes, and options snapshots.
7. Persist bars, metrics, regimes, snapshots, and signals into SQLite.
8. Feed replay, research, daily review, paper trading, dashboards, and verification tooling from the same persisted state.

The Kalshi side runs a related but distinct loop:

1. Discover tradable Kalshi BTC strike markets.
2. Maintain market metadata and order books.
3. Maintain a BTC truth-price window from external spot feeds.
4. Estimate the fair probability that the settlement average lands above or below a strike.
5. Compare fair probability to the order book to compute edge.
6. Apply strategy gates, risk controls, and per-market sizing.
7. Emit paper or live signals, manage fills, and track bot-level performance.
8. Run a population / farm layer that evaluates many parameterized bots on the same shared market state.

## Runtime Entry Points

### Core Argus

Run the core orchestrator with:

```powershell
python main.py
```

`main.py` just parses an optional log level and calls `src.orchestrator.main()`.

The core orchestrator is the central runtime coordinator. It loads config and secrets, opens the SQLite databases, starts the event bus, wires up connectors, bar building, feature generation, regime detection, detectors, paper trading, dashboards, soak tooling, and the optional governance / agent layers.

### Kalshi

Run the Kalshi stack with:

```powershell
python -m argus_kalshi
```

By default this uses `config/kalshi_farm_compact.yaml` plus `config/secrets.yaml`. You can override both with `--settings` and `--secrets`, or run only the terminal UI with `--ui-only`.

## Configuration and Secrets

The core loaders expect these files:

- `config/config.yaml`
- `config/thresholds.yaml`
- `config/secrets.yaml`

The Kalshi side expects:

- a settings YAML such as `config/kalshi_farm_compact.yaml`
- `config/secrets.yaml`

If your clone does not ship populated config files, create them manually under `config/` before running the system. Secrets should never be committed. The loaders support `ARGUS_SECRETS` and the Kalshi CLI supports `ARGUS_SECRETS_PATH`.

At minimum, operators usually need some subset of:

- Alpaca credentials
- Tastytrade session or OAuth credentials
- Public.com API secret and account ID if using Public options
- Telegram bot credentials if alerts are enabled
- Kalshi key ID and private-key path for the Kalshi side
- Optional Alpha Vantage, NewsAPI, Reddit, or other service credentials if those modules are enabled

## Core Argus Architecture

### 1. Event Bus

The core runtime is event-driven. Connectors publish typed events such as quotes, bars, metrics, and signals. Downstream components subscribe to topics and do not mutate upstream state. This keeps ingestion, analytics, persistence, and research loosely coupled.

The practical consequence is that a provider can change without changing the downstream consumers, as long as it emits the canonical event types.

### 2. Bar Builder

`src/core/bar_builder.py` consumes `market.quotes` and emits 1-minute bars aligned to UTC minute boundaries.

Key rules:

- It uses exchange timestamps, not local arrival time, for bar alignment.
- It rejects quotes with missing or invalid source timestamps.
- It floors timestamps to the start of the UTC minute:

```text
minute_open = floor(timestamp_seconds / 60) * 60
```

- It discards late ticks for already-closed bars.
- It treats cumulative 24-hour exchange volume as a cumulative counter and uses deltas:

```text
volume_delta_t = max(0, cumulative_volume_t - cumulative_volume_{t-1})
```

That prevents grossly over-counting bar volume.

### 3. Feature Builder

`src/core/feature_builder.py` computes rolling market features from bars.

Implemented metrics:

- Log return:

```text
r_t = ln(C_t / C_{t-1})
```

- Realized volatility over a rolling window:

```text
sigma_realized = sqrt(sample_variance(r_{t-n+1} ... r_t)) * annualization_factor
```

For 1-minute bars, the annualization factor is `sqrt(365.25 * 24 * 60)`.

- Jump score:

```text
jump_score_t = |r_t| / sigma_window
```

This is a simple standardized-move detector. Large absolute returns relative to recent volatility become high jump scores.

### 4. Regime Detector

`src/core/regime_detector.py` is the main deterministic regime classifier.

Per symbol, it maintains:

- EMA fast and EMA slow
- RSI
- ATR
- Rolling realized volatility
- Spread and volume histories
- Gap flags and warmup state

Core math and heuristics:

- Trend is inferred from EMA slope and relative placement of fast and slow EMA.
- Volatility regime is based on volatility z-scores and threshold bands.
- ATR normalizes move size by recent range.
- Liquidity regime uses either quote spread or bar-proxy spread:

```text
spread_pct = (ask - bid) / mid
```

or, if quotes are unavailable:

```text
spread_proxy = (high - low) / ((high + low) / 2)
```

- Gaps are detected by comparing actual bar timestamps to expected timestamps at the configured bar duration.

The module also supports:

- hysteresis and minimum dwell periods
- warmup decay after data gaps
- quote-based liquidity instead of bar-only proxies
- optional market-wide risk basket aggregation

### 5. Market Regime and Global Risk Flow

The codebase includes a market-level scaffold and a concrete global-risk-flow signal.

`src/core/global_risk_flow.py` computes:

```text
GlobalRiskFlow = 0.4 * AsiaReturn + 0.4 * EuropeReturn + 0.2 * FXRiskSignal
```

Where:

- `AsiaReturn` is the mean daily return of `EWJ, FXI, EWT, EWY, INDA`
- `EuropeReturn` is the mean daily return of `EWG, EWU, FEZ, EWL`
- `FXRiskSignal` is the daily return of `USD/JPY`

Only completed daily bars strictly before the simulation time are used, so replay avoids lookahead.

### 6. News Sentiment

The news sentiment path combines RSS ingestion and lexicon scoring. The lexicon scorer uses finance-specific positive and negative word lists and tokenizes headline text into words.

The simplest sentiment score is effectively:

```text
score = positive_matches - negative_matches
```

with normalized reporting fields such as word count and per-headline aggregation. This is intentionally transparent and deterministic rather than model-heavy.

### 7. Options Snapshots, Greeks, and IV

The options layer is a major part of the current codebase.

Main pieces:

- `src/connectors/public_options.py`
- `src/connectors/tastytrade_options.py`
- `src/connectors/tastytrade_streamer.py`
- `src/analysis/greeks_engine.py`
- `src/core/iv_consensus.py`

What they do:

- Public.com provides snapshot-based option Greeks and chain structure.
- Tastytrade provides nested chain snapshots and live DXLink quote / Greeks streaming.
- Snapshots are normalized into a single event schema.
- The Greeks engine uses a European Black-Scholes approximation for individual options and spreads.

Black-Scholes quantities used by the engine are the standard:

```text
d1 = [ln(S / K) + (r + sigma^2 / 2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

The engine then derives Delta, Gamma, Theta, Vega, and Rho from the closed-form model. It also solves implied volatility numerically with Brent's method when a provider IV is missing and the quote is liquid enough.

Illiquid quotes are screened before IV inversion:

- zero or missing bid
- spread too wide relative to mid
- premium too close to zero

This is designed to avoid manufacturing unstable IV from bad quotes.

### 8. Detectors and Strategy Logic

The repository contains multiple detector and strategy modules under `src/detectors/` and `src/strategies/`.

Representative logic includes:

- volatility and options-IV detectors
- ETF options detector
- VRP-style credit spread logic
- overnight session strategy
- regime-conditional and router layers

Most of these modules combine some subset of:

- regime filters
- volatility thresholds
- spread / liquidity gates
- PoP or expectancy thresholds
- DTE and strike-structure constraints

The exact thresholds come from config and threshold files, but the structural idea is consistent: signals are generated only when state, volatility, and data quality line up.

### 9. Paper Trading and the Paper Trader Farm

There are two distinct paper-trading concepts in the codebase.

`src/analysis/paper_trader.py` handles individual virtual ETF spread trades. It records:

- entry credit
- spread width
- contracts
- entry Greeks
- exit debit
- PnL in dollars and percent

The spread PnL logic is the standard credit-spread payoff:

```text
PnL = entry_credit - exit_debit
```

with status tracking for profit exits, loss exits, time exits, or expiration.

`src/trading/paper_trader_farm.py` scales this idea into a large parameterized farm. It generates many trader configs, converts config vectors into tensor form for batch evaluation, and applies:

- trade-universe guards
- economic-calendar blackout checks
- drawdown circuit breakers
- per-minute trade-rate limits
- per-symbol and per-farm position caps

This is effectively a combinatorial search / simulation harness around the same signal stream.

### 10. Replay Packs and Backtesting

`src/tools/replay_pack.py` slices:

- market bars
- forward outcomes
- symbol / market regimes
- option chain snapshots

for a symbol and date range into a deterministic replay artifact.

Replay is designed to be strict:

- only completed data is visible at a simulated timestamp
- no future snapshots leak backward
- derived IV is only used when provider IV is missing and quote data is available

This makes the replay system a bridge between live ingestion and offline research.

### 11. Research Loop and Evaluator

The research stack includes:

- experiment runners
- strategy evaluators
- Monte Carlo / bootstrap metrics
- data-snooping controls

`src/analysis/strategy_evaluator.py` builds a deterministic composite score from metrics such as:

- total PnL
- Sharpe proxy
- max drawdown
- expectancy
- profit factor
- fill rate
- regime-conditioned performance

It also adds penalties for:

- excessive drawdown
- high reject rate
- parameter fragility
- regime concentration
- walk-forward instability

#### Reality Check

`src/analysis/reality_check.py` implements White's Reality Check with a stationary bootstrap. The idea is to test whether the best strategy is genuinely better than a benchmark after correcting for data snooping across many tested variants.

The stationary bootstrap draws random-length blocks with expected block size `b`:

```text
P(start new block) = 1 / b
```

This preserves dependence structure better than naive IID resampling.

#### Deflated Sharpe Ratio

`src/analysis/deflated_sharpe.py` implements the Deflated Sharpe Ratio from Bailey and Lopez de Prado. It adjusts an observed Sharpe ratio for:

- multiple testing / selection bias
- skewness
- kurtosis

The output is a probability-like score used as a deployment-quality gate rather than just a ranking number.

### 12. Dashboard, Soak, and Governance Layers

The repository also contains:

- a web dashboard under `src/dashboard/web.py`
- soak-test tools under `src/soak/`
- a governance / agent layer under `src/agent/`

These modules do not change the basic market math. They sit around the runtime to provide:

- observability
- long-run stability checks
- tape capture
- operational governance
- tool-mediated orchestration

## Argus Kalshi Architecture

The Kalshi stack is implemented under `argus_kalshi/` and is more specialized than the core Argus runtime.

### 1. Market Discovery and Shared State

`argus_kalshi/kalshi_markets.py` and related modules discover tradable Kalshi BTC strike contracts and classify them by asset, settlement time, and family. Shared state is used heavily so many bots can evaluate the same market state without duplicating expensive subscriptions.

### 2. Truth Feed and BTC Window Engine

The Kalshi strategy depends on the settlement definition for BTC strike contracts. The truth-feed path collects recent BTC mid prices and builds a rolling 60-second window, because the contract resolves on the simple average over that window rather than a single last trade.

The window engine keeps:

- the last 60 seconds of observed prices
- the sum of those prices
- the current simple average

That lets the probability engine reason about both fully future and partially observed settlement windows.

### 3. Probability Engine

`argus_kalshi/kalshi_probability.py` is one of the most mathematically explicit parts of the codebase.

The model assumes short-horizon geometric Brownian motion with near-zero drift:

```text
dS / S = sigma dW
```

Two cases are handled.

#### Case A: settlement window fully in the future

If time to settlement is greater than 60 seconds, the 60-second average is entirely unobserved. The code approximates:

```text
E[A] ~= S_now
Var[A] ~= S_now^2 * sigma^2 * T_avg / 3
P(YES) = Phi((E[A] - K) / sqrt(Var[A]))
```

where:

- `A` is the future settlement-window average
- `S_now` is current BTC mid price
- `K` is the strike
- `Phi` is the standard normal CDF

#### Case B: already inside the settlement window

If some of the 60-second averaging window has already occurred, then a portion of the average is locked in. Let:

```text
n_obs = observed seconds
tau = remaining seconds
S_obs = sum of observed prices
R = 60*K - S_obs
m_req = R / tau
```

The remaining path must produce an average high enough to clear the strike after accounting for the already observed prices. The code then approximates the remaining-average distribution and computes the probability of clearing `m_req`.

#### Volatility and Tail Adjustments

The module estimates volatility from recent log returns:

```text
r_t = ln(S_t / S_{t-1})
sigma = sqrt(sample_variance(r)) / sqrt(dt)
```

It also includes:

- excess kurtosis estimation for tail heaviness
- a HAR-J style jump-aware volatility estimator
- short-horizon momentum estimation via OLS slope on log prices

That means the final fair probability is not just a naive fixed-vol diffusion number; it can be made heavier-tailed and drift-aware.

### 4. Strategy Engine

`argus_kalshi/kalshi_strategy.py` converts fair probabilities and order books into trade decisions.

Core edge math:

```text
EV_yes = p_yes - ask_yes_prob
EV_no  = (1 - p_yes) - ask_no_prob
```

If one side clears the configured edge threshold and the relevant risk gates are green, the strategy emits a signal.

Risk filters include:

- per-market exposure caps
- daily drawdown halts
- WebSocket health checks
- order-book validity checks
- truth-feed staleness checks
- cooldown and persistence windows

There are also family and asset-level allowlists to avoid classes of contracts that have historically performed poorly in paper results.

### 5. Mispricing Scalper

`argus_kalshi/mispricing_scalper.py` is a shorter-horizon directional engine. It does not rely only on model-fair versus ask. It builds a directional composite score from:

- drift from the probability engine
- order-book imbalance
- depth pressure
- trade-flow imbalance
- order-delta flow on both YES and NO books

Conceptually:

```text
score = w1 * drift
      + w2 * orderbook_imbalance
      + w3 * trade_flow
      + w4 * depth_pressure
      + w5 * delta_yes_flow
      - w6 * delta_no_flow
```

The actual weights are config-driven. The point is that the scalper is microstructure-driven, not just diffusion-model-driven.

### 6. Execution Engine and Order Management

The Kalshi execution path consumes trade signals, respects bot IDs, and routes orders through Kalshi REST and WebSocket state. The system tracks fills, position updates, account balance, and market-side caps. It is designed so signals can be computed even when live trading is disabled, which is useful for paper and audit modes.

### 7. Farm Runner and Population Logic

`argus_kalshi/farm_runner.py` is the large-scale population system.

It runs many isolated bots over the same shared market data and groups them into cohorts by strategy parameters such as:

- edge thresholds
- persistence windows
- entry price bounds
- cooldowns
- max quantity
- family / asset enablement
- crowding throttles
- momentum and flow thresholds

Supporting modules include:

- `farm_grid.py`
- `population_scaler.py`
- `simulation.py`
- `edge_tracker.py`
- `context_policy.py`
- `regime_gate.py`

These modules are effectively a population-search and adaptive-allocation layer. They use shared state, context keys, robustness scoring, novelty distance, and family weighting to decide which bot families deserve promotion or demotion.

### 8. Terminal UI and IPC

The Kalshi side also has a dedicated terminal UI and IPC channel so the trading process and visualization process can be separated. This is operationally useful because the UI can connect to the strategy process without embedding all of the rendering concerns inside the execution loop.

## Algorithms and Math by Major Subsystem

This section summarizes the major algorithmic pieces in one place.

| Subsystem | Core method |
| --- | --- |
| Bar building | UTC minute bucketing, immutable close-on-rollover, cumulative-volume delta extraction |
| Feature building | Log returns, rolling sample variance, annualized realized volatility, standardized jump score |
| Regime detection | EMA trend state, RSI, ATR, rolling vol, spread / liquidity bands, gap-state heuristics |
| Global risk flow | Weighted average of Asia, Europe, and FX daily-return components |
| News sentiment | Tokenization plus positive / negative lexicon counting |
| Greeks engine | European Black-Scholes plus Brent IV inversion |
| Replay | Strict as-of slicing with no lookahead |
| Strategy evaluator | Composite weighted score plus penalties for fragility and instability |
| Reality check | White's Reality Check with stationary bootstrap |
| Deflated Sharpe | Selection-bias- and non-normality-adjusted Sharpe significance |
| Kalshi probability | GBM-based approximation for 60-second settlement average crossing a strike |
| Kalshi strategy | Fair probability versus order-book implied price edge |
| Kalshi scalper | Weighted microstructure directional score |
| Farm layers | Parameter-grid exploration, shared-state evaluation, family weighting, and adaptive promotion |

## Typical Workflows

### Core runtime

```powershell
python main.py
```

### Kalshi runtime

```powershell
python -m argus_kalshi
```

### Kalshi UI only

```powershell
python -m argus_kalshi --ui-only --connect 127.0.0.1:9999
```

### Verification and health

Useful commands already present in the repo:

```powershell
python scripts\verify_argus.py --quick
python scripts\verify_argus.py --data --symbol SPY --days 5
python scripts\verify_argus.py --replay --symbol SPY --days 5
python scripts\verify_system.py
python scripts\e2e_verify.py
python scripts\verify_vrp_replay.py --symbol SPY --start 2026-03-01 --end 2026-03-05
pytest
```

## Practical Notes and Constraints

- The repo is designed around deterministic event processing and replayability.
- Core Argus defaults to collector mode unless explicitly placed in live mode.
- The core and Kalshi stacks share some infrastructure ideas, but they are not the same program and should be understood as related systems inside one repository.
- Config and secrets are part of the runtime contract. A clone without populated config files will need those files created before the programs can run.
- Some modules are intentionally conservative and prefer skipping bad data over manufacturing unstable state.

## Summary

Argus is not just a data collector. It is a full pipeline from provider policy selection, event normalization, deterministic bar construction, and regime classification through options analytics, replay, research ranking, and paper trading. The `argus_kalshi/` side extends the repository into event-driven prediction-market trading with a dedicated probability model for Kalshi BTC strike contracts, a directional scalper, and a large shared-state bot farm.

If you are reading the code for the first time, start with:

1. `main.py`
2. `src/orchestrator.py`
3. `src/core/bar_builder.py`
4. `src/core/feature_builder.py`
5. `src/core/regime_detector.py`
6. `src/tools/replay_pack.py`
7. `argus_kalshi/__main__.py`
8. `argus_kalshi/runner.py`
9. `argus_kalshi/kalshi_probability.py`
10. `argus_kalshi/kalshi_strategy.py`

Those files give the clearest top-down view of how the repository works today.
