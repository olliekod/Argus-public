# Created by Oliver Meihls

# Pantheon Context Loader
#
# Provides one-call bootstrap functions that populate a :class:`ContextInjector`
# from live Argus data sources.
#
# Called once at :class:`~scripts.research_engine.ResearchEngine` startup and
# refreshed between research cases.
#
# Data sources:
# - ``data/argus.db``    — symbol universe (bar availability), experiment results
# - ``data/factory.db``  — promoted strategy library
# - ``config/``          — risk engine hard limits

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Individual Loaders

def load_available_symbols(
    db_path: str = "data/argus.db",
) -> List[str]:
    # Query argus.db for all symbols that have actual bar data.
    #
    # Returns a deduplicated, sorted list of confirmed tradeable tickers.
    # Returns empty list gracefully if the DB or table does not exist.
    path = Path(db_path)
    if not path.exists():
        logger.debug("context_loader: argus.db not found at %s — symbol universe empty", db_path)
        return []

    try:
        conn = sqlite3.connect(path)
        # Try the canonical bar tables (market_bars, or bars if aliased)
        for table in ("market_bars", "bars", "bar_data"):
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            if cur.fetchone() is not None:
                rows = conn.execute(
                    f"SELECT DISTINCT symbol FROM {table} WHERE symbol IS NOT NULL"
                ).fetchall()
                conn.close()
                symbols = sorted({r[0].upper() for r in rows if r[0]})
                logger.info("context_loader: loaded %d symbols from %s.%s", len(symbols), db_path, table)
                return symbols
        conn.close()
        logger.debug("context_loader: no bar table found in %s", db_path)
        return []
    except Exception as exc:
        logger.warning("context_loader: failed to load symbol universe: %s", exc)
        return []


def load_strategy_library(
    factory_db: str = "data/factory.db",
    max_entries: int = 30,
) -> List[Dict[str, Any]]:
    # Query factory.db for all promoted strategies.
    #
    # Returns a list of dicts with keys: name, grading, universe, signals.
    # Used by Prometheus (novelty) and Athena (novelty scoring).
    path = Path(factory_db)
    if not path.exists():
        logger.debug("context_loader: factory.db not found at %s — strategy library empty", factory_db)
        return []

    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        # Only return promoted strategies ordered by best confidence first
        rows = conn.execute(
            """
            SELECT name, grading, final_manifest
            FROM strategies
            WHERE status = 'PROMOTE'
            ORDER BY athena_confidence DESC
            LIMIT ?
            """,
            (max_entries,),
        ).fetchall()
        conn.close()

        entries = []
        for row in rows:
            manifest: Dict[str, Any] = {}
            if row["final_manifest"]:
                try:
                    manifest = json.loads(row["final_manifest"])
                except Exception:
                    pass

            entries.append({
                "name": row["name"],
                "grading": row["grading"],
                "universe": manifest.get("universe", []),
                "signals": manifest.get("signals", []),
                "category": _infer_category(manifest),
            })

        logger.info("context_loader: loaded %d promoted strategies from factory.db", len(entries))
        return entries
    except Exception as exc:
        logger.warning("context_loader: failed to load strategy library: %s", exc)
        return []


def load_hades_performance_log(
    argus_db: str = "data/argus.db",
    max_entries: int = 10,
) -> List[Dict[str, Any]]:
    # Query the experiment_results table for recent backtest outcomes.
    #
    # Returns a list of dicts with: strategy_name, sharpe, pnl, win_rate,
    # kill_reason, grading.
    #
    # Returns empty list gracefully if the table doesn't exist (Hades hasn't
    # run yet, or different schema).
    path = Path(argus_db)
    if not path.exists():
        return []

    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row

        # Check for experiment_results table
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_results'"
        )
        if cur.fetchone() is None:
            conn.close()
            logger.debug("context_loader: experiment_results table not found in argus.db")
            return []

        rows = conn.execute(
            """
            SELECT strategy_id, sharpe_annualized_proxy, total_realized_pnl,
                   win_rate, kill_reason, grading
            FROM experiment_results
            ORDER BY run_ts DESC
            LIMIT ?
            """,
            (max_entries,),
        ).fetchall()
        conn.close()

        entries = []
        for row in rows:
            entries.append({
                "strategy_name": str(row["strategy_id"] or "unknown"),
                "sharpe": float(row["sharpe_annualized_proxy"] or 0.0),
                "pnl": float(row["total_realized_pnl"] or 0.0),
                "win_rate": float(row["win_rate"] or 0.0),
                "kill_reason": row["kill_reason"] or None,
                "grading": row["grading"] or "Unrated",
            })

        logger.info("context_loader: loaded %d Hades results from argus.db", len(entries))
        return entries
    except Exception as exc:
        logger.warning("context_loader: failed to load Hades performance log: %s", exc)
        return []


def load_risk_limits(config: Dict[str, Any]) -> Dict[str, Any]:
    # Extract key risk caps from the loaded Argus config dict.
    #
    # Returns a flat dict of the most operationally relevant hard limits
    # for Prometheus to respect when sizing strategies.
    risk = config.get("risk_engine", {})
    limits: Dict[str, Any] = {}

    # Try common key variants across config schema versions
    for key, aliases in {
        "max_risk_per_trade_pct": ["max_risk_per_trade_pct", "risk_per_trade_pct", "max_risk_pct"],
        "aggregate_cap_pct":      ["aggregate_cap_pct", "max_exposure_pct", "total_exposure_cap"],
        "max_contracts":          ["max_contracts_per_play", "max_contracts", "position_max_contracts"],
        "max_loss_per_contract":  ["max_loss_per_contract", "max_contract_loss", "loss_cap_per_contract"],
        "max_drawdown_pct":       ["drawdown_throttle_pct", "max_drawdown_pct", "drawdown_limit"],
    }.items():
        for alias in aliases:
            val = risk.get(alias)
            if val is not None:
                limits[key] = val
                break

    # Sensible defaults if not configured (these are conservative hard limits)
    limits.setdefault("max_risk_per_trade_pct", 0.02)   # 2% per trade
    limits.setdefault("aggregate_cap_pct", 0.20)         # 20% total exposure
    limits.setdefault("max_drawdown_pct", 0.15)          # 15% drawdown trigger

    logger.debug("context_loader: risk limits loaded: %s", limits)
    return limits


# Master Bootstrap

def bootstrap_context_injector(
    config: Optional[Dict[str, Any]] = None,
    argus_db: str = "data/argus.db",
    factory_db: str = "data/factory.db",
) -> "ContextInjector":  # type: ignore[name-defined]
    # Build a fully-enriched ContextInjector from all live Argus data sources.
    #
    # Called once at ResearchEngine startup. Safe to call repeatedly — all
    # loaders handle missing tables / DBs gracefully and return empty lists.
    #
    # Parameters
    # config : dict, optional
    # Loaded Argus config dict (from ``load_config()``). Used for risk limits.
    # If None, conservative defaults are applied.
    # argus_db : str
    # Path to argus.db (symbol universe + Hades results).
    # factory_db : str
    # Path to factory.db (promoted strategy library).
    #
    # Returns
    # ContextInjector
    # Fully populated context injector ready for use in the debate pipeline.
    from src.agent.pantheon.roles import ContextInjector

    ctx = ContextInjector()

    # 1. Indicator descriptions (auto-loaded from HADES_INDICATOR_DESCRIPTIONS default)
    #    Already set by default_factory in ContextInjector dataclass.

    # 2. Symbol universe
    symbols = load_available_symbols(argus_db)
    if symbols:
        ctx.set_available_symbols(symbols)

    # 3. Risk limits
    cfg = config or {}
    ctx.set_risk_limits(load_risk_limits(cfg))

    # 4. Strategy library
    for entry in load_strategy_library(factory_db):
        ctx.add_strategy_to_library(
            name=entry["name"],
            grading=entry["grading"],
            universe=entry["universe"],
            signals=entry["signals"],
            category=entry.get("category", ""),
        )

    # 5. Hades performance log
    for entry in load_hades_performance_log(argus_db):
        ctx.add_hades_result(
            strategy_name=entry["strategy_name"],
            sharpe=entry["sharpe"],
            pnl=entry["pnl"],
            win_rate=entry["win_rate"],
            kill_reason=entry.get("kill_reason"),
            grading=entry.get("grading", "Unrated"),
        )

    logger.info(
        "context_loader: bootstrap complete — %d symbols, %d strategies, %d Hades results",
        len(symbols),
        len(ctx.strategy_library),
        len(ctx.hades_performance_log),
    )
    return ctx


# Helpers

def _infer_category(manifest: Dict[str, Any]) -> str:
    # Infer a rough strategy category from the manifest for library display.
    name = manifest.get("name", "").lower()
    signals = [s.lower() for s in manifest.get("signals", [])]
    direction = manifest.get("direction", "").upper()

    if "vrp" in name or "put spread" in name or "call spread" in name or direction == "NEUTRAL":
        return "VRP/Options"
    if "momentum" in name or "ema" in signals or "macd" in signals:
        return "Momentum"
    if "mean reversion" in name or "rsi" in signals or "bollinger" in signals:
        return "Mean Reversion"
    if "breakout" in name or "ema_slope" in signals:
        return "Breakout"
    if "vol" in name or "rolling_vol" in signals or "vol_z" in signals:
        return "Volatility"
    return "Other"
