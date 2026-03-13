# Argus

Argus is a real-time market data ingestion and strategy research platform. It collects live data from exchanges and brokers, turns it into bars, detects market regimes, and generates trading signals for manual execution. No trades are placed automatically; it runs in collector mode by default.

## What It Does:

Argus pulls market data from multiple sources: crypto perpetuals via Bybit, options IV via Deribit, and ETFs (IBIT, BITO + liquid ETF universe: SPY, QQQ, IWM, DIA, TLT, GLD, XLF, XLK, XLE, SMH) via Alpaca and Yahoo. Raw ticks are normalized and aggregated into 1-minute OHLCV bars. A feature builder computes returns, volatility, and jump scores. A regime detector classifies each symbol and the overall market (trend, volatility state, session). Detectors look for specific setups (e.g., BTC IV spike plus IBIT drawdown) and emit signals. A paper trader farm runs hundreds of thousands of virtual traders with different parameter sets to evaluate which strategies would have performed well.

Determinism is central. Same tape input produces same bars, same regimes, same signals. No randomness or wall-clock dependency in the core pipeline. A tape recorder captures events for replay and backtesting.

## Data Flow:

Connectors publish quotes and metrics to an event bus. The bar builder subscribes to quotes, aggregates ticks into 1-minute bars, and publishes bar events. The feature builder and regime detector consume bars and emit metrics and regime labels. Detectors and strategies consume bars plus regimes and emit signals. Persistence writes bars and signals to SQLite. The paper trader farm and Telegram bot consume signals.

## Connectors:

- Bybit (crypto perpetuals, WebSocket)
- Deribit (BTC options IV)
- Alpaca (IBIT, BITO equity and bars)
- Yahoo Finance (equity data)
- Polymarket (optional, disabled by default)

## Detectors and Strategies:

- IBIT detector: sell put spreads when BTC volatility spikes and IBIT drops
- Options IV detector: implied volatility setups
- Volatility detector: regime shifts and expansion events
- Day-of-week + regime timing gate: filter signals based on session, trend, volatility, and data quality

## Indicators:

Deterministic implementations of EMA, RSI, VWAP, MACD, ATR, rolling volatility, and log returns. Supports batch and incremental modes.

## Modes:

Collector mode (default): collects data, builds bars, detects regimes, generates signals. No execution.

Live mode: set `ARGUS_MODE=live` to enable trading. Not recommended until strategies are validated.

## Setup:

Requires Python 3.10+, API keys for Bybit and Alpaca (Deribit optional for IV data), and a Telegram bot token if you want alerts.

```powershell
cd C:\Users\Oliver\Desktop\Desktop\Projects\argus
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config\secrets.example.yaml config\secrets.yaml
# Secrets.yaml need to have updated api keys
python scripts\init_database.py
python main.py
```

## Configuration

Main config lives in `config/config.yaml`. Thresholds and strategy params in `config/thresholds.yaml` and `config/strategy_params.json`. Secrets (API keys, Telegram token) go in `config/secrets.yaml` (gitignored). Use `config/secrets.example.yaml` as the starting template.

### Regime detector robustness toggles (backward-compatible defaults)

All of the following are configured via `RegimeDetector` thresholds and are **off / neutral by default** so existing strategy behavior is unchanged unless you opt in:

- Hysteresis + dwell: `vol_hysteresis_enabled`, `vol_hysteresis_band`, `trend_hysteresis_enabled`, `trend_hysteresis_slope_band`, `trend_hysteresis_strength_band`, `min_dwell_bars`.
- Gap-aware warmth/confidence: `gap_confidence_decay_threshold_ms`, `gap_confidence_decay_multiplier`, `gap_warmth_decay_bars`, `gap_reset_window_threshold_ms`.
- Quote-based liquidity spread: `quote_liquidity_enabled` (when true, spread uses latest quote snapshot with `recv_ts_ms <= asof_ts`; otherwise bar proxy is used).
- Trend acceleration metric: persisted in regime metrics as `trend_accel`; classification impact remains disabled unless `trend_accel_classification_enabled=true`.

### Market risk regime scaffold

A `MarketRegimeDetector` scaffold is available and can be enabled with `system.risk_basket_symbols` in `config/config.yaml` (default empty/disabled).

- Example: `risk_basket_symbols: ["SPY", "TLT", "GLD"]`
- Emits global market events on `regimes.market` with `risk_regime` and `metrics_json`.
- If basket symbols are missing, emits `UNKNOWN`.

## Tastytrade OAuth Bootstrap (one-time)

Prerequisites: add your OAuth client credentials to `config/secrets.yaml`:

```yaml
tastytrade_oauth2:
  client_id: "<CLIENT_ID>"
  client_secret: "<CLIENT_SECRET>"
```

Run the bootstrap helper:

```powershell
python scripts\tastytrade_oauth_bootstrap.py
```

This opens (or prints) `http://127.0.0.1:8777/oauth/tastytrade/start`, which redirects you to the Tastytrade consent screen. After approval, you are sent back to `/oauth/tastytrade/callback` and the refresh token is saved to `config/secrets.yaml` under `tastytrade_oauth2.refresh_token`.

To re-run the flow (for example after revoking credentials), simply run the helper again and complete the browser step.


## Liquid ETF Universe Operations

Universe tickers: `SPY, QQQ, IWM, DIA, TLT, GLD, XLF, XLK, XLE, SMH`.

Verification and audits:

```bash
python scripts/verify_system.py
python scripts/verify_system.py --deep
python scripts/tastytrade_health_audit.py --symbol SPY --quotes --duration 15 --json-out logs/spy.json
python scripts/tastytrade_health_audit.py --universe --quotes --duration 10 --json-out logs/universe.json
python scripts/provider_benchmark.py --duration 10 --json-out logs/bench.json
python scripts/prune_option_snapshots.py --days 14
```

Storage policy: Argus persists sampled options quote snapshots (deterministic slice) and does **not** persist full-chain option ticks.

## Pre-commit secrets guard

To prevent accidental secret commits, enable the repo hook:

```powershell
copy scripts\precommit_check_secrets.py .git\hooks\pre-commit
```

You can also run the check manually:

```powershell
python scripts\precommit_check_secrets.py
```

## Rules

90-day rule: after adopting a strategy, no parameter changes for 90 days.

Circuit breakers: auto-pause on 5% daily loss or 5 consecutive losses.

Observation first: run in collector mode and validate before considering live execution.

## License

Private project. Not for redistribution.
