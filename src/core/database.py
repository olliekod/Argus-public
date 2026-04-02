# Created by Oliver Meihls

# Argus Database Module
#
# SQLite database for storing detections, market data, and system health.
# Uses aiosqlite for async operations.

import asyncio
import aiosqlite
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Database:
    # Async SQLite database manager for Argus.
    #
    # Handles:
    # - Detection logging
    # - Funding rate history
    # - Options IV data
    # - Liquidation events
    # - Price snapshots
    # - System health monitoring
    
    def __init__(self, db_path: str):
        # Initialize database connection.
        #
        # Args:
        # db_path: Path to SQLite database file
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
        self._write_lock: Optional[asyncio.Lock] = None
    
    async def connect(self) -> None:
        # Establish database connection and create tables.
        self._connection = await aiosqlite.connect(str(self.db_path))
        self._write_lock = asyncio.Lock()
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        # Enable WAL mode for long-running app and concurrent readers
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
        # Set WAL size limit to 512MB (prevents unlimited growth between checkpoints)
        await self._connection.execute("PRAGMA journal_size_limit=536870912")
        # Wait up to 30s on lock (avoids OperationalError: database is locked when another process/connection holds the DB)
        await self._connection.execute("PRAGMA busy_timeout=30000")
        logger.info(f"Database connected: {self.db_path}")

    async def checkpoint(self) -> None:
        # Perform a WAL checkpoint to truncate the -wal file.
        if self._connection and self._write_lock:
            try:
                # RESTART: Truncates the WAL file and resets it.
                # Must be done when no other connections have open transactions.
                async with self._write_lock:
                    await self._connection.execute("PRAGMA wal_checkpoint(RESTART)")
                logger.info("Database WAL checkpoint (RESTART) completed")
            except Exception as e:
                logger.error(f"Failed to perform WAL checkpoint: {e}")

    async def close(self) -> None:
        # Close database connection.
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    async def execute(self, sql: str, params: tuple = ()) -> None:
        # Execute a SQL statement (for external use).
        if self._write_lock is None:
            raise RuntimeError("Database is not connected")
        async with self._write_lock:
            await self._connection.execute(sql, params)
            await self._connection.commit()

    async def execute_many(self, sql: str, params_list: List[tuple]) -> None:
        # Execute a SQL statement with multiple parameter sets in a single transaction.
        if self._write_lock is None:
            raise RuntimeError("Database is not connected")
        async with self._write_lock:
            await self._connection.executemany(sql, params_list)
            await self._connection.commit()

    async def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        # Fetch one row from a query.
        cursor = await self._connection.execute(sql, params)
        return await cursor.fetchone()
    
    async def fetch_all(self, sql: str, params: tuple = ()) -> List[tuple]:
        # Fetch all rows from a query.
        cursor = await self._connection.execute(sql, params)
        return await cursor.fetchall()
    
    async def _create_tables(self) -> None:
        # Create all database tables if they don't exist.

        # Followed traders list (for best-trader follow feature)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS followed_traders (
                trader_id TEXT PRIMARY KEY,
                followed_at TEXT NOT NULL,
                score REAL,
                scoring_method TEXT,
                window_days INTEGER,
                config_json TEXT
            )
        """)

        # Signals table (order intent / signal log)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                trader_id TEXT,
                strategy_type TEXT,
                symbol TEXT NOT NULL,
                direction TEXT,
                signal_source TEXT,
                iv_at_signal REAL,
                warmth_at_signal REAL,
                pop_at_signal REAL,
                gap_risk_at_signal REAL,
                underlying_price REAL,
                btc_price REAL,
                btc_iv REAL,
                conditions_score INTEGER,
                conditions_label TEXT,
                strikes TEXT,
                expiry TEXT,
                target_credit REAL,
                bid_price REAL,
                ask_price REAL,
                spread_width REAL,
                contracts INTEGER,
                outcome TEXT,
                outcome_reason TEXT
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_ts ON trade_signals(timestamp)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_trader ON trade_signals(trader_id)"
        )

        # Phase 4: Kalshi Outcomes (for trade history and PnL)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS kalshi_outcomes (
                market_ticker TEXT NOT NULL,
                bot_id TEXT NOT NULL DEFAULT 'default',
                strike REAL,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                outcome TEXT,   -- 'WON' or 'LOST'
                final_avg REAL,
                settled_at_ms INTEGER,
                is_paper BOOLEAN DEFAULT 1,
                PRIMARY KEY (market_ticker, bot_id)
            )
        """)

        # Phase 4: Kalshi Terminal Events
        # Safe migration for pre-bot_id Kalshi outcomes table.
        cur = await self._connection.execute("PRAGMA table_info(kalshi_outcomes)")
        table_info = await cur.fetchall()
        await cur.close()
        cols = {row[1] for row in table_info}
        pk_cols = [row[1] for row in table_info if row[5] > 0]
        needs_pk_migration = pk_cols == ["market_ticker"]
        if "bot_id" not in cols or needs_pk_migration:
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS kalshi_outcomes_new (
                    market_ticker TEXT NOT NULL,
                    bot_id TEXT NOT NULL DEFAULT 'default',
                    strike REAL,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    outcome TEXT,
                    final_avg REAL,
                    settled_at_ms INTEGER,
                    is_paper BOOLEAN DEFAULT 1,
                    PRIMARY KEY (market_ticker, bot_id)
                )
            """)
            if "bot_id" in cols:
                await self._connection.execute("""
                    INSERT OR REPLACE INTO kalshi_outcomes_new (
                        market_ticker, bot_id, strike, direction, entry_price, exit_price,
                        quantity, pnl, outcome, final_avg, settled_at_ms, is_paper
                    )
                    SELECT market_ticker, bot_id, strike, direction, entry_price, exit_price,
                           quantity, pnl, outcome, final_avg, settled_at_ms, is_paper
                    FROM kalshi_outcomes
                """)
            else:
                await self._connection.execute("""
                    INSERT OR REPLACE INTO kalshi_outcomes_new (
                        market_ticker, bot_id, strike, direction, entry_price, exit_price,
                        quantity, pnl, outcome, final_avg, settled_at_ms, is_paper
                    )
                    SELECT market_ticker, 'default', strike, direction, entry_price, exit_price,
                           quantity, pnl, outcome, final_avg, settled_at_ms, is_paper
                    FROM kalshi_outcomes
                """)
            await self._connection.execute("DROP TABLE kalshi_outcomes")
            await self._connection.execute("ALTER TABLE kalshi_outcomes_new RENAME TO kalshi_outcomes")
            await self._connection.commit()

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS kalshi_terminal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                bot_id TEXT NOT NULL DEFAULT 'default'
            )
        """)

        # Migration for kalshi_terminal_events and kalshi_decisions (bot_id)
        for table, col in [("kalshi_terminal_events", "bot_id"), ("kalshi_decisions", "bot_id")]:
            try:
                cur = await self._connection.execute(f"PRAGMA table_info({table})")
                info = await cur.fetchall()
                await cur.close()
                if not any(row[1] == col for row in info):
                    await self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT 'default'")
            except Exception:
                pass

        # Phase 4: Kalshi Strategy Decisions
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS kalshi_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                p_yes REAL,
                yes_ask INTEGER,
                no_ask INTEGER,
                action_taken TEXT NOT NULL,
                reason TEXT,
                bot_id TEXT NOT NULL DEFAULT 'default'
            )
        """)

        # Uniformity monitor snapshots
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS uniformity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_type TEXT,
                variable_name TEXT NOT NULL,
                unique_count INTEGER,
                total_count INTEGER,
                modal_value TEXT,
                modal_pct REAL,
                hhi REAL,
                entropy REAL,
                is_alert INTEGER DEFAULT 0,
                alert_reason TEXT
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_uniformity_ts ON uniformity_snapshots(timestamp)"
        )

        # Daily trader metrics (for per-trader PnL analytics)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS trader_daily_metrics (
                trader_id TEXT NOT NULL,
                date TEXT NOT NULL,
                realized_pnl REAL DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                trades_opened INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                peak_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                PRIMARY KEY (trader_id, date)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tdm_date ON trader_daily_metrics(date)"
        )

        # Main detections table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                opportunity_type TEXT NOT NULL,
                asset TEXT NOT NULL,
                exchange TEXT NOT NULL,
                current_price REAL,
                volume_24h REAL,
                volatility_1h REAL,
                volatility_24h REAL,
                detection_data TEXT,
                estimated_edge_bps REAL,
                estimated_slippage_bps REAL,
                estimated_fees_bps REAL,
                net_edge_bps REAL,
                would_trigger_entry INTEGER DEFAULT 0,
                suggested_position_size REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                resolution_timestamp TEXT,
                actual_outcome TEXT,
                hypothetical_pnl_percent REAL,
                hypothetical_pnl_usd REAL,
                alert_tier INTEGER,
                alert_sent INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        # Funding rates history
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                funding_rate REAL NOT NULL,
                predicted_rate REAL,
                open_interest REAL,
                volume_24h REAL,
                mark_price REAL,
                index_price REAL
            )
        """)
        
        # Options IV history
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS options_iv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                expiry TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                implied_volatility REAL NOT NULL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                mark_price REAL,
                underlying_price REAL
            )
        """)
        
        # Liquidation events
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS liquidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                side TEXT NOT NULL,
                liquidation_amount_usd REAL NOT NULL,
                price REAL NOT NULL
            )
        """)
        
        # Price snapshots
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                price_type TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL
            )
        """)
        
        # System health
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                latency_ms REAL
            )
        """)
        
        # Daily statistics rollup
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_detections INTEGER,
                detections_by_type TEXT,
                total_hypothetical_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl_percent REAL,
                total_pnl_usd REAL,
                best_trade_pnl REAL,
                worst_trade_pnl REAL,
                avg_trade_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL
            )
        """)
        
        # Create indices for common queries
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(opportunity_type)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_asset ON detections(asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_funding_timestamp ON funding_rates(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_options_timestamp ON options_iv(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_liquidations_timestamp ON liquidations(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON price_snapshots(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp)"
        )
        
        await self._connection.commit()

        # Paper equity epochs (for reset_paper command)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS paper_equity_epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch_start TEXT NOT NULL,
                starting_equity REAL NOT NULL,
                reason TEXT,
                scope TEXT DEFAULT 'all'
            )
        """)

        # 1-minute OHLCV bars (from BarBuilder → PersistenceManager)
        # Unique constraint: (source, symbol, bar_duration, timestamp) for idempotent upserts
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS market_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                tick_count INTEGER DEFAULT 0,
                n_ticks INTEGER DEFAULT 0,
                first_source_ts REAL,
                last_source_ts REAL,
                late_ticks_dropped INTEGER DEFAULT 0,
                close_reason INTEGER DEFAULT 0,
                bar_duration INTEGER DEFAULT 60,
                UNIQUE(source, symbol, bar_duration, timestamp)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_bars_ts_sym "
            "ON market_bars(timestamp, symbol)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_bars_source_sym_ts "
            "ON market_bars(source, symbol, timestamp)"
        )

        # ── Provenance column migration (safe ALTER for existing DBs) ──
        for col, col_type, default in [
            ("n_ticks", "INTEGER", "0"),
            ("first_source_ts", "REAL", "NULL"),
            ("last_source_ts", "REAL", "NULL"),
            ("late_ticks_dropped", "INTEGER", "0"),
            ("close_reason", "INTEGER", "0"),
            ("bar_duration", "INTEGER", "60"),
        ]:
            try:
                await self._connection.execute(
                    f"ALTER TABLE market_bars ADD COLUMN {col} {col_type} DEFAULT {default}"
                )
            except Exception:
                pass  # column already exists

        # Component heartbeat log
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS component_heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                uptime_seconds REAL,
                events_processed INTEGER DEFAULT 0,
                latest_lag_ms REAL,
                health TEXT DEFAULT 'ok',
                extra_json TEXT
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_heartbeats_ts_comp "
            "ON component_heartbeats(timestamp, component)"
        )

        # Signal events (from detectors → PersistenceManager)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                detector TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                data TEXT
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_ev_ts "
            "ON signal_events(timestamp)"
        )

        # Generic market metrics (Funding, IV, Correlation, etc.)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS market_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL,
                metadata_json TEXT,
                UNIQUE(timestamp, source, symbol, metric)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_query "
            "ON market_metrics(symbol, metric, timestamp)"
        )

    async def get_metrics(
        self, 
        symbol: str, 
        metric: Optional[str] = None,
        start_ms: Optional[int] = None, 
        end_ms: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        # Fetch historical metrics for a symbol and date range.
        query = "SELECT timestamp, source, symbol, metric, value, metadata_json FROM market_metrics WHERE symbol = ?"
        params = [symbol]
        
        if metric:
            query += " AND metric = ?"
            params.append(metric)
        if start_ms:
            # market_metrics uses text ISO timestamps, so we might need conversion or just string compare
            # but wait, the table definition uses TEXT. 
            # I'll use BETWEEN if start/end are provided as ISO strings or do conversion.
            pass

        # Actually, let's use the timestamp text comparison if possible, or convert ms to ISO.
        def _ms_to_iso(ms: int) -> str:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace('+00:00', 'Z')

        if start_ms:
            query += " AND timestamp >= ?"
            params.append(_ms_to_iso(start_ms))
        if end_ms:
            query += " AND timestamp <= ?"
            params.append(_ms_to_iso(end_ms))
            
        query += " ORDER BY timestamp ASC"
        
        cursor = await self._connection.execute(query, tuple(params))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
        # Safe migration: add columns if they don't exist yet (idempotent)
        for col, col_type in [
            ("event_type", "TEXT"),
            ("scope", "TEXT"),
            ("timeframe", "INTEGER"),
            ("timestamp", "INTEGER"),
            ("config_hash", "TEXT"),
            ("vol_regime", "TEXT"),
            ("trend_regime", "TEXT"),
            ("liquidity_regime", "TEXT"),
            ("session_regime", "TEXT"),
            ("risk_regime", "TEXT"),
            ("spread_pct", "REAL"),
            ("volume_pctile", "REAL"),
            ("confidence", "REAL"),
            ("is_warm", "INTEGER"),
        ]:
            try:
                await self._connection.execute(
                    f"ALTER TABLE regimes ADD COLUMN {col} {col_type}"
                )
            except Exception: pass

        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_regimes_lookup "
            "ON regimes(event_type, scope, timeframe, timestamp DESC)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_regimes_scope_ts "
            "ON regimes(scope, timestamp)"
        )

        # Phase 3: Trading signals (from strategy modules)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                timestamp_ms INTEGER NOT NULL,
                strategy_id TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                entry_type TEXT,
                entry_price REAL,
                stop_price REAL,
                tp_price REAL,
                horizon TEXT,
                confidence REAL,
                quality_score INTEGER,
                data_quality_flags INTEGER,
                regime_snapshot_json TEXT,
                features_snapshot_json TEXT,
                explain TEXT,
                case_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add case_id to signals if missing
        try:
            await self._connection.execute(
                "ALTER TABLE signals ADD COLUMN case_id TEXT"
            )
        except Exception:
            pass  # column already exists
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_lookup "
            "ON signals(strategy_id, symbol, timestamp_ms DESC)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_time "
            "ON signals(timestamp_ms DESC)"
        )

        # Phase 3: Signal outcomes (markouts)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                timestamp_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                case_id TEXT,
                ret_1bar REAL,
                ret_5bar REAL,
                ret_10bar REAL,
                ret_60bar REAL,
                pnl_1bar REAL,
                pnl_5bar REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add case_id to signal_outcomes if missing
        try:
            await self._connection.execute(
                "ALTER TABLE signal_outcomes ADD COLUMN case_id TEXT"
            )
        except Exception:
            pass  # column already exists
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_lookup "
            "ON signal_outcomes(strategy_id, timestamp_ms DESC)"
        )

        # Phase 3B: Option contracts (static metadata)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS option_contracts (
                contract_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                option_symbol TEXT NOT NULL UNIQUE,
                strike REAL NOT NULL,
                expiration_ms INTEGER NOT NULL,
                option_type TEXT NOT NULL,
                multiplier INTEGER DEFAULT 100,
                style TEXT DEFAULT 'american',
                provider TEXT NOT NULL,
                first_seen_ms INTEGER NOT NULL,
                last_updated_ms INTEGER NOT NULL
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_oc_symbol_exp "
            "ON option_contracts(symbol, expiration_ms)"
        )

        # Phase 3B: Option quotes (time series)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS option_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                expiration_ms INTEGER NOT NULL,
                option_type TEXT NOT NULL,
                bid REAL,
                ask REAL,
                last REAL,
                mid REAL,
                volume INTEGER,
                open_interest INTEGER,
                iv REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                timestamp_ms INTEGER NOT NULL,
                source_ts_ms INTEGER,
                recv_ts_ms INTEGER,
                provider TEXT,
                FOREIGN KEY (contract_id) REFERENCES option_contracts(contract_id)
            )
        """)
        # Safe migration: add columns if they don't exist yet (idempotent)
        for col, col_type in [
            ("contract_id", "TEXT"),
            ("symbol", "TEXT"),
            ("strike", "REAL"),
            ("expiration_ms", "INTEGER"),
            ("option_type", "TEXT"),
            ("timestamp_ms", "INTEGER"),
            ("recv_ts_ms", "INTEGER"),
        ]:
            try:
                await self._connection.execute(
                    f"ALTER TABLE option_quotes ADD COLUMN {col} {col_type}"
                )
            except Exception: pass

        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_oq_ts "
            "ON option_quotes(timestamp_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_oq_recv_ts "
            "ON option_quotes(recv_ts_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_oq_symbol_exp "
            "ON option_quotes(symbol, expiration_ms)"
        )

        # Phase 3B: Option chain snapshots (atomic snapshots)
        # Unique constraint: (provider, symbol, timestamp_ms)
        # - Prevents same-provider duplicates; Alpaca and Tastytrade can both
        #   have a row at the same timestamp_ms (different provider).
        # - Connectors use poll-time floored to minute for timestamp_ms so
        #   granularity is consistent; recv_ts_ms stores actual receipt time.
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                expiration_ms INTEGER NOT NULL,
                underlying_price REAL NOT NULL,
                n_strikes INTEGER,
                atm_iv REAL,
                timestamp_ms INTEGER NOT NULL,
                source_ts_ms INTEGER,
                recv_ts_ms INTEGER,
                provider TEXT NOT NULL,
                quotes_json TEXT NOT NULL,
                UNIQUE(provider, symbol, timestamp_ms)
            )
        """)
        # Safe migration: add columns if they don't exist yet (idempotent)
        for col, col_type in [
            ("symbol", "TEXT"),
            ("expiration_ms", "INTEGER"),
            ("timestamp_ms", "INTEGER"),
            ("recv_ts_ms", "INTEGER"),
            ("provider", "TEXT"),
        ]:
            try:
                await self._connection.execute(
                    f"ALTER TABLE option_chain_snapshots ADD COLUMN {col} {col_type}"
                )
            except Exception: pass

        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocs_ts "
            "ON option_chain_snapshots(timestamp_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocs_recv_ts "
            "ON option_chain_snapshots(recv_ts_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocs_symbol_recv "
            "ON option_chain_snapshots(symbol, recv_ts_ms)"
        )
        # Ensure UNIQUE(provider, symbol, timestamp_ms) exists for ON CONFLICT (migration for DBs created before this constraint)
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ocs_provider_symbol_ts "
            "ON option_chain_snapshots(provider, symbol, timestamp_ms)"
        )
        # Migrate away from old UNIQUE(symbol, expiration_ms, timestamp_ms) so multiple providers can store
        await self._migrate_option_chain_snapshots_if_old_schema()

        # Phase 4A.1: Bar outcomes (backtest ground truth)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS bar_outcomes (
                provider TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bar_duration_seconds INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                outcome_version TEXT NOT NULL,
                
                -- Core metrics (quantized REAL)
                close_now REAL,
                close_at_horizon REAL,
                fwd_return REAL,
                max_runup REAL,
                max_drawdown REAL,
                realized_vol REAL,
                
                -- Path helpers (for exit simulation)
                max_high_in_window REAL,
                min_low_in_window REAL,
                max_runup_ts_ms INTEGER,
                max_drawdown_ts_ms INTEGER,
                time_to_max_runup_ms INTEGER,
                time_to_max_drawdown_ms INTEGER,
                
                -- Coverage & debug
                status TEXT NOT NULL,
                close_ref_ms INTEGER,
                window_start_ms INTEGER,
                window_end_ms INTEGER,
                bars_expected INTEGER,
                bars_found INTEGER,
                gap_count INTEGER,
                computed_at_ms INTEGER,
                
                PRIMARY KEY (provider, symbol, bar_duration_seconds, timestamp_ms, horizon_seconds, outcome_version)
            ) WITHOUT ROWID
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_bar_outcomes_lookup "
            "ON bar_outcomes(provider, symbol, bar_duration_seconds, timestamp_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_bar_outcomes_status "
            "ON bar_outcomes(status, provider, symbol)"
        )

        # Phase 4A.1+: System heartbeat for uptime tracking
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS system_heartbeat (
                component TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                metadata_json TEXT,
                PRIMARY KEY (component, timestamp_ms)
            ) WITHOUT ROWID
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sys_hb_comp_ts "
            "ON system_heartbeat(component, timestamp_ms)"
        )

        await self._connection.commit()

        logger.debug("Database tables created/verified")

# Phase 3B: Options Persistence (idempotent upserts)

    async def upsert_option_contract(
        self,
        contract_id: str,
        symbol: str,
        option_symbol: str,
        strike: float,
        expiration_ms: int,
        option_type: str,
        provider: str,
        timestamp_ms: int,
        multiplier: int = 100,
        style: str = "american",
    ) -> bool:
        # Upsert option contract metadata (idempotent).
        #
        # Unique constraint: option_symbol (globally unique OCC symbol).
        #
        # Returns:
        # True if inserted, False if updated existing.
        try:
            await self._connection.execute("""
                INSERT INTO option_contracts (
                    contract_id, symbol, option_symbol, strike, expiration_ms,
                    option_type, multiplier, style, provider, first_seen_ms, last_updated_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(option_symbol) DO UPDATE SET
                    last_updated_ms = excluded.last_updated_ms
            """, (
                contract_id, symbol, option_symbol, strike, expiration_ms,
                option_type, multiplier, style, provider, timestamp_ms, timestamp_ms,
            ))
            await self._connection.commit()
            return True
        except Exception as e:
            logger.warning("upsert_option_contract failed: %s", e)
            return False

    async def upsert_option_chain_snapshot(
        self,
        snapshot_id: str,
        symbol: str,
        expiration_ms: int,
        underlying_price: float,
        n_strikes: int,
        atm_iv: float | None,
        timestamp_ms: int,
        source_ts_ms: int | None,
        recv_ts_ms: int | None,
        provider: str,
        quotes_json: str,
    ) -> bool:
        # Upsert option chain snapshot (idempotent).
        #
        # Unique constraint: (provider, symbol, timestamp_ms).
        # Expiration is stored in payload, not in uniqueness key.
        #
        # Returns:
        # True if inserted, False if already exists.
        try:
            await self._connection.execute("""
                INSERT INTO option_chain_snapshots (
                    snapshot_id, symbol, expiration_ms, underlying_price,
                    n_strikes, atm_iv, timestamp_ms, source_ts_ms, recv_ts_ms,
                    provider, quotes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, symbol, timestamp_ms) DO NOTHING
            """, (
                snapshot_id, symbol, expiration_ms, underlying_price,
                n_strikes, atm_iv, timestamp_ms, source_ts_ms, recv_ts_ms,
                provider, quotes_json,
            ))
            await self._connection.commit()
            return True
        except Exception as e:
            logger.warning("upsert_option_chain_snapshot failed: %s", e)
            return False

    async def get_latest_chain_timestamp(self, symbol: str) -> int | None:
        # Get latest chain snapshot timestamp for a symbol.
        #
        # Used for restart initialization to avoid duplicate polling.
        #
        # Returns:
        # Latest timestamp_ms or None if no snapshots exist.
        cursor = await self._connection.execute("""
            SELECT MAX(timestamp_ms) FROM option_chain_snapshots WHERE symbol = ?
        """, (symbol,))
        row = await cursor.fetchone()
        return row[0] if row and row[0] else None

    async def _migrate_option_chain_snapshots_if_old_schema(self) -> None:
        # If the table has old UNIQUE(symbol, expiration_ms, timestamp_ms), recreate with UNIQUE(provider, symbol, timestamp_ms).
        cursor = await self._connection.execute(
            "PRAGMA index_list('option_chain_snapshots')"
        )
        rows = await cursor.fetchall()
        has_old_constraint = False
        for row in rows:
            # (seq, name, unique)
            if not row[2]:  # not unique
                continue
            idx_name = row[1]
            info_cur = await self._connection.execute(
                f"PRAGMA index_info('{idx_name}')"
            )
            info = await info_cur.fetchall()
            cols = {r[2] for r in info}  # column names
            if cols == {"symbol", "expiration_ms", "timestamp_ms"}:
                has_old_constraint = True
                break
        if not has_old_constraint:
            return
        logger.info(
            "Migrating option_chain_snapshots from UNIQUE(symbol, expiration_ms, timestamp_ms) to UNIQUE(provider, symbol, timestamp_ms)"
        )
        await self._connection.execute("""
            CREATE TABLE option_chain_snapshots_new (
                snapshot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                expiration_ms INTEGER NOT NULL,
                underlying_price REAL NOT NULL,
                n_strikes INTEGER,
                atm_iv REAL,
                timestamp_ms INTEGER NOT NULL,
                source_ts_ms INTEGER,
                recv_ts_ms INTEGER,
                provider TEXT NOT NULL,
                quotes_json TEXT NOT NULL,
                UNIQUE(provider, symbol, timestamp_ms)
            )
        """)
        await self._connection.execute("""
            INSERT OR IGNORE INTO option_chain_snapshots_new
            (snapshot_id, symbol, expiration_ms, underlying_price, n_strikes, atm_iv,
             timestamp_ms, source_ts_ms, recv_ts_ms, provider, quotes_json)
            SELECT snapshot_id, symbol, expiration_ms, underlying_price, n_strikes, atm_iv,
                   timestamp_ms, source_ts_ms, recv_ts_ms, provider, quotes_json
            FROM option_chain_snapshots
            ORDER BY timestamp_ms DESC
        """)
        await self._connection.execute("DROP TABLE option_chain_snapshots")
        await self._connection.execute(
            "ALTER TABLE option_chain_snapshots_new RENAME TO option_chain_snapshots"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocs_ts ON option_chain_snapshots(timestamp_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocs_recv_ts ON option_chain_snapshots(recv_ts_ms)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocs_symbol_recv ON option_chain_snapshots(symbol, recv_ts_ms)"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ocs_provider_symbol_ts ON option_chain_snapshots(provider, symbol, timestamp_ms)"
        )
        await self._connection.commit()
        logger.info("option_chain_snapshots migration completed")

    
    # Detection Operations
    
    async def insert_detection(self, detection: Dict[str, Any]) -> int:
        # Insert a new detection record.
        #
        # Args:
        # detection: Detection data dictionary
        #
        # Returns:
        # ID of inserted record
        detection_data = detection.get('detection_data')
        if isinstance(detection_data, dict):
            try:
                detection_data = json.dumps(detection_data)
            except TypeError:
                detection_data = json.dumps(detection_data, default=str)
        
        cursor = await self._connection.execute("""
            INSERT INTO detections (
                timestamp, opportunity_type, asset, exchange,
                current_price, volume_24h, volatility_1h, volatility_24h,
                detection_data, estimated_edge_bps, estimated_slippage_bps,
                estimated_fees_bps, net_edge_bps, would_trigger_entry,
                suggested_position_size, entry_price, stop_loss, take_profit,
                alert_tier, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.get('timestamp', datetime.now(timezone.utc).isoformat()),
            detection.get('opportunity_type'),
            detection.get('asset'),
            detection.get('exchange'),
            detection.get('current_price'),
            detection.get('volume_24h'),
            detection.get('volatility_1h'),
            detection.get('volatility_24h'),
            detection_data,
            detection.get('estimated_edge_bps'),
            detection.get('estimated_slippage_bps'),
            detection.get('estimated_fees_bps'),
            detection.get('net_edge_bps'),
            1 if detection.get('would_trigger_entry') else 0,
            detection.get('suggested_position_size'),
            detection.get('entry_price'),
            detection.get('stop_loss'),
            detection.get('take_profit'),
            detection.get('alert_tier'),
            detection.get('notes')
        ))
        
        await self._connection.commit()
        logger.debug(f"Inserted detection: {detection.get('opportunity_type')} - {detection.get('asset')}")
        return cursor.lastrowid

    async def insert_kalshi_outcome(
        self,
        market_ticker: str,
        strike: float,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        pnl: float,
        outcome: str,
        final_avg: float,
        settled_at_ms: int,
        is_paper: bool = True,
        bot_id: str = "default",
    ) -> bool:
        # Insert a Kalshi trade outcome (settlement result).
        try:
            if self._write_lock is None:
                raise RuntimeError("Database is not connected")
            async with self._write_lock:
                await self._connection.execute("""
                    INSERT OR REPLACE INTO kalshi_outcomes (
                        market_ticker, bot_id, strike, direction, entry_price, exit_price,
                        quantity, pnl, outcome, final_avg, settled_at_ms, is_paper
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_ticker, bot_id, strike, direction, entry_price, exit_price,
                    quantity, pnl, outcome, final_avg, settled_at_ms, int(is_paper)
                ))
                await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting Kalshi outcome: {e}")
            return False

    async def get_kalshi_daily_pnl(self, days: int = 30, is_paper: bool = True, bot_id: str = "default") -> List[Dict[str, Any]]:
        # Get daily Kalshi PnL totals for the last N days.
        # Convert settled_at_ms to YYYY-MM-DD
        query = """
            SELECT 
                date(settled_at_ms / 1000, 'unixepoch') as date,
                SUM(pnl) as daily_pnl,
                COUNT(*) as trade_count
            FROM kalshi_outcomes
            WHERE is_paper = ? AND bot_id = ?
            GROUP BY date
            ORDER BY date DESC
            LIMIT ?
        """
        cursor = await self._connection.execute(query, (int(is_paper), bot_id, days))
        rows = await cursor.fetchall()
        return [{"date": row["date"], "pnl": row["daily_pnl"], "daily_pnl": row["daily_pnl"], "trade_count": row["trade_count"]} for row in rows]

    async def get_kalshi_daily_pnl_history(self, days: int = 30, is_paper: bool = True, bot_id: str = "default") -> List[Dict[str, Any]]:
        # Return daily Kalshi PnL time series filtered by bot_id.
        return await self.get_kalshi_daily_pnl(days=days, is_paper=is_paper, bot_id=bot_id)

    async def get_kalshi_outcome_stats(self) -> dict:
        # Return all-time PnL and win rate from kalshi_outcomes (for UI).
        try:
            if self._connection is None:
                return {"total_pnl": 0.0, "wins": 0, "total": 0}
            cursor = await self._connection.execute(
                """SELECT
                    COALESCE(SUM(pnl), 0) AS total_pnl,
                    SUM(CASE WHEN outcome = 'WON' THEN 1 ELSE 0 END) AS wins,
                    COUNT(*) AS total
                   FROM kalshi_outcomes"""
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return {"total_pnl": 0.0, "wins": 0, "total": 0}
            return {
                "total_pnl": float(row[0]) if row[0] is not None else 0.0,
                "wins": int(row[1]) if row[1] is not None else 0,
                "total": int(row[2]) if row[2] is not None else 0,
            }
        except Exception as e:
            logger.warning("get_kalshi_outcome_stats failed: %s", e)
            return {"total_pnl": 0.0, "wins": 0, "total": 0}

    async def update_detection_resolution(
        self,
        detection_id: int,
        outcome: str,
        pnl_percent: float,
        pnl_usd: float
    ) -> None:
        # Update a detection with resolution data.
        #
        # Args:
        # detection_id: ID of detection to update
        # outcome: 'normalized', 'stopped', 'profit_taken', 'expired'
        # pnl_percent: Hypothetical P&L percentage
        # pnl_usd: Hypothetical P&L in USD
        await self._connection.execute("""
            UPDATE detections
            SET resolution_timestamp = ?,
                actual_outcome = ?,
                hypothetical_pnl_percent = ?,
                hypothetical_pnl_usd = ?
            WHERE id = ?
        """, (
            datetime.now(timezone.utc).isoformat(),
            outcome,
            pnl_percent,
            pnl_usd,
            detection_id
        ))
        await self._connection.commit()
        logger.debug(f"Updated detection {detection_id}: {outcome}, {pnl_percent:.2f}%")
    
    async def mark_alert_sent(self, detection_id: int) -> None:
        # Mark a detection's alert as sent.
        await self._connection.execute(
            "UPDATE detections SET alert_sent = 1 WHERE id = ?",
            (detection_id,)
        )
        await self._connection.commit()
    
    async def get_recent_detections(
        self,
        hours: int = 24,
        opportunity_type: Optional[str] = None
    ) -> List[Dict]:
        # Get recent detections.
        #
        # Args:
        # hours: How many hours back to look
        # opportunity_type: Filter by type (optional)
        #
        # Returns:
        # List of detection dictionaries
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        if opportunity_type:
            cursor = await self._connection.execute("""
                SELECT * FROM detections
                WHERE timestamp > ? AND opportunity_type = ?
                ORDER BY timestamp DESC
            """, (cutoff, opportunity_type))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM detections
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def get_unresolved_detections(self) -> List[Dict]:
        # Get detections that would trigger entry but haven't been resolved.
        cursor = await self._connection.execute("""
            SELECT * FROM detections
            WHERE would_trigger_entry = 1
              AND resolution_timestamp IS NULL
            ORDER BY timestamp DESC
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
# Funding Rate Operations
    
    async def insert_funding_rate(
        self,
        exchange: str,
        asset: str,
        funding_rate: float,
        **kwargs
    ) -> None:
        # Insert a funding rate record.
        await self._connection.execute("""
            INSERT INTO funding_rates (
                timestamp, exchange, asset, funding_rate,
                predicted_rate, open_interest, volume_24h, mark_price, index_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs.get('timestamp', datetime.now(timezone.utc).isoformat()),
            exchange,
            asset,
            funding_rate,
            kwargs.get('predicted_rate'),
            kwargs.get('open_interest'),
            kwargs.get('volume_24h'),
            kwargs.get('mark_price'),
            kwargs.get('index_price')
        ))
        await self._connection.commit()
    
    async def get_funding_history(
        self,
        asset: str,
        periods: int = 30,
        exchange: Optional[str] = None
    ) -> List[Dict]:
        # Get recent funding rate history.
        #
        # Args:
        # asset: Asset symbol
        # periods: Number of periods to retrieve
        # exchange: Filter by exchange (optional)
        #
        # Returns:
        # List of funding rate records
        if exchange:
            cursor = await self._connection.execute("""
                SELECT * FROM funding_rates
                WHERE asset = ? AND exchange = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (asset, exchange, periods))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM funding_rates
                WHERE asset = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (asset, periods))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
# Options IV Operations
    
    async def insert_options_iv(
        self,
        asset: str,
        expiry: str,
        strike: float,
        option_type: str,
        iv: float,
        **kwargs
    ) -> None:
        # Insert an options IV record.
        await self._connection.execute("""
            INSERT INTO options_iv (
                timestamp, asset, expiry, strike, option_type,
                implied_volatility, delta, gamma, theta, vega,
                mark_price, underlying_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs.get('timestamp', datetime.now(timezone.utc).isoformat()),
            asset,
            expiry,
            strike,
            option_type,
            iv,
            kwargs.get('delta'),
            kwargs.get('gamma'),
            kwargs.get('theta'),
            kwargs.get('vega'),
            kwargs.get('mark_price'),
            kwargs.get('underlying_price')
        ))
        await self._connection.commit()
    
    async def get_iv_history(
        self,
        asset: str,
        days: int = 30
    ) -> List[Dict]:
        # Get IV history for an asset.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = await self._connection.execute("""
            SELECT * FROM options_iv
            WHERE asset = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (asset, cutoff))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
# Liquidation Operations
    
    async def insert_liquidation(
        self,
        exchange: str,
        asset: str,
        side: str,
        amount_usd: float,
        price: float,
        timestamp: Optional[str] = None
    ) -> None:
        # Insert a liquidation event.
        await self._connection.execute("""
            INSERT INTO liquidations (timestamp, exchange, asset, side, liquidation_amount_usd, price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp or datetime.now(timezone.utc).isoformat(),
            exchange,
            asset,
            side,
            amount_usd,
            price
        ))
        await self._connection.commit()
    
    async def get_recent_liquidations(
        self,
        asset: str,
        minutes: int = 5
    ) -> List[Dict]:
        # Get recent liquidations for cascade detection.
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        cursor = await self._connection.execute("""
            SELECT * FROM liquidations
            WHERE asset = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (asset, cutoff))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
# Price Snapshot Operations
    
    async def insert_price_snapshot(
        self,
        exchange: str,
        asset: str,
        price_type: str,
        price: float,
        volume: Optional[float] = None
    ) -> None:
        # Insert a price snapshot.
        await self._connection.execute("""
            INSERT INTO price_snapshots (timestamp, exchange, asset, price_type, price, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            exchange,
            asset,
            price_type,
            price,
            volume
        ))
        await self._connection.commit()
    
    async def get_price_history(
        self,
        asset: str,
        exchange: str,
        price_type: str = 'spot',
        minutes: int = 60
    ) -> List[float]:
        # Get price history for volatility calculations.
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        cursor = await self._connection.execute("""
            SELECT price FROM price_snapshots
            WHERE asset = ? AND exchange = ? AND price_type = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (asset, exchange, price_type, cutoff))
        rows = await cursor.fetchall()
        return [row['price'] for row in rows]
    
# System Health Operations
    
    async def insert_health_check(
        self,
        component: str,
        status: str,
        error_message: Optional[str] = None,
        latency_ms: Optional[float] = None
    ) -> None:
        # Record a system health check.
        await self._connection.execute("""
            INSERT INTO system_health (timestamp, component, status, error_message, latency_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            component,
            status,
            error_message,
            latency_ms
        ))
        await self._connection.commit()
    
    async def get_component_status(self, component: str) -> Optional[Dict]:
        # Get the most recent status for a component.
        cursor = await self._connection.execute("""
            SELECT * FROM system_health
            WHERE component = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (component,))
        row = await cursor.fetchone()
        return dict(row) if row else None

# Market Metrics Operations

    async def insert_market_metric(
        self,
        timestamp: str,
        source: str,
        symbol: str,
        metric: str,
        value: float,
        metadata_json: Optional[str] = None
    ) -> None:
        # Insert a generic market metric record (deduplicated).
        await self._connection.execute("""
            INSERT OR IGNORE INTO market_metrics (
                timestamp, source, symbol, metric, value, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, source, symbol, metric, value, metadata_json))
        await self._connection.commit()

    async def get_latest_timestamps(self, tables: List[str]) -> Dict[str, Optional[str]]:
        # Fetch the most recent timestamp for each table.
        results: Dict[str, Optional[str]] = {}
        for table in tables:
            cursor = await self._connection.execute(
                f"SELECT MAX(timestamp) AS latest FROM {table}"
            )
            row = await cursor.fetchone()
            results[table] = row["latest"] if row and row["latest"] else None
        return results

    async def get_price_at_or_before(
        self,
        exchange: str,
        asset: str,
        price_type: str,
        cutoff_timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        # Fetch the most recent price at or before a cutoff timestamp.
        cursor = await self._connection.execute("""
            SELECT price, timestamp
            FROM price_snapshots
            WHERE exchange = ? AND asset = ? AND price_type = ?
              AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (exchange, asset, price_type, cutoff_timestamp))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_per_trader_pnl(
        self,
        days: int = 30,
        min_trades: int = 1,
        epoch_start: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Get per-trader realized PnL statistics for /pnl analytics.
        #
        # Computes per-trader return as realized_pnl / starting_equity.
        # Each trader starts with the same notional ($5000), so return = total_pnl / 5000.
        params: List[Any] = [f"-{days} days", min_trades]
        epoch_clause = ""
        if epoch_start:
            epoch_clause = " AND timestamp >= ?"
            params.insert(1, epoch_start)
        cursor = await self._connection.execute(f"""
            SELECT
                trader_id,
                strategy_type,
                COUNT(*) as total_trades,
                SUM(CASE WHEN status != 'open' THEN 1 ELSE 0 END) as closed_trades,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl <= 0 AND realized_pnl IS NOT NULL THEN 1 ELSE 0 END) as losses,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                MIN(realized_pnl) as worst_trade,
                MAX(realized_pnl) as best_trade
            FROM paper_trades
            WHERE timestamp >= datetime('now', ?)
            {epoch_clause}
            GROUP BY trader_id, strategy_type
            HAVING closed_trades >= ?
        """, tuple(params))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_zombie_positions(self, stale_hours: int = 48) -> List[Dict[str, Any]]:
        # Find zombie positions: open trades with no updates for stale_hours.
        #
        # A zombie position is an open position that is no longer reachable by the
        # strategy lifecycle (e.g., missing close event, orphan order, expired option
        # not settled, process crash before DB update).
        #
        # Detection rules:
        # 1. Status = 'open' AND timestamp < (now - stale_hours)
        # 2. Status = 'open' AND expiry date is in the past
        cursor = await self._connection.execute("""
            SELECT id, trader_id, strategy_type, symbol, timestamp,
                   strikes, expiry, entry_credit, contracts, status,
                   market_conditions
            FROM paper_trades
            WHERE status = 'open'
              AND (
                  timestamp < datetime('now', ?)
                  OR (expiry IS NOT NULL AND expiry < date('now'))
              )
            ORDER BY timestamp ASC
        """, (f"-{stale_hours} hours",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def mark_zombies(self, trade_ids: List[str], reason: str = 'zombie_detected') -> int:
        # Mark a list of trades as zombie (expired with reason).
        if not trade_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ','.join(['?'] * len(trade_ids))
        await self._connection.execute(f"""
            UPDATE paper_trades
            SET status = 'expired',
                close_timestamp = ?,
                market_conditions = json_set(
                    COALESCE(market_conditions, '{{}}'),
                    '$.zombie_reason', ?
                )
            WHERE id IN ({placeholders})
              AND status = 'open'
        """, (now, reason, *trade_ids))
        await self._connection.commit()
        return len(trade_ids)

    async def get_followed_traders(self) -> List[Dict[str, Any]]:
        # Get the list of followed traders.
        cursor = await self._connection.execute(
            "SELECT * FROM followed_traders ORDER BY score DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def set_followed_traders(self, traders: List[Dict[str, Any]]) -> None:
        # Replace the followed traders list.
        await self._connection.execute("DELETE FROM followed_traders")
        for t in traders:
            await self._connection.execute("""
                INSERT INTO followed_traders
                (trader_id, followed_at, score, scoring_method, window_days, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                t['trader_id'],
                t.get('followed_at', datetime.now(timezone.utc).isoformat()),
                t.get('score'),
                t.get('scoring_method'),
                t.get('window_days'),
                t.get('config_json'),
            ))
        await self._connection.commit()

    async def get_trader_performance(self, days: int = 60) -> List[Dict[str, Any]]:
        # Get aggregated trader performance over a window.
        cursor = await self._connection.execute("""
            SELECT 
                trader_id,
                strategy_type,
                COUNT(*) as total_trades,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM paper_trades
            WHERE timestamp >= datetime('now', ?)
            GROUP BY trader_id, strategy_type
        """, (f"-{days} days",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
# Statistics Operations
    
    async def get_statistics(self, days: int = 14) -> Dict:
        # Get aggregated statistics for analysis.
        #
        # Returns dict with:
        # - total_detections
        # - detections_by_type
        # - trade_statistics
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        # Total detections
        cursor = await self._connection.execute("""
            SELECT COUNT(*) as count FROM detections WHERE timestamp > ?
        """, (cutoff,))
        row = await cursor.fetchone()
        total = row['count'] if row else 0
        
        # By type
        cursor = await self._connection.execute("""
            SELECT opportunity_type, COUNT(*) as count
            FROM detections
            WHERE timestamp > ?
            GROUP BY opportunity_type
        """, (cutoff,))
        rows = await cursor.fetchall()
        by_type = {row['opportunity_type']: row['count'] for row in rows}
        
        # Trade statistics
        cursor = await self._connection.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN hypothetical_pnl_percent > 0 THEN 1 ELSE 0 END) as winners,
                AVG(hypothetical_pnl_percent) as avg_pnl,
                MIN(hypothetical_pnl_percent) as worst_pnl,
                MAX(hypothetical_pnl_percent) as best_pnl
            FROM detections
            WHERE timestamp > ?
              AND would_trigger_entry = 1
              AND resolution_timestamp IS NOT NULL
        """, (cutoff,))
        row = await cursor.fetchone()
        
        trade_stats = {
            'total_trades': row['total_trades'] or 0,
            'winners': row['winners'] or 0,
            'avg_pnl': row['avg_pnl'] or 0,
            'worst_pnl': row['worst_pnl'] or 0,
            'best_pnl': row['best_pnl'] or 0,
        }
        
        if trade_stats['total_trades'] > 0:
            trade_stats['win_rate'] = trade_stats['winners'] / trade_stats['total_trades']
        else:
            trade_stats['win_rate'] = 0
        
        return {
            'total_detections': total,
            'detections_by_type': by_type,
            'trade_statistics': trade_stats
        }
    
# Maintenance Operations
    
    async def cleanup_old_data(self, retention_days: Dict[str, int]) -> None:
        # Delete old data based on retention policy.
        #
        # Args:
        # retention_days: Dict mapping table names to retention days.
        # Special keys:
        # - ``"option_chain_snapshots"`` — uses ``timestamp_ms``
        # (epoch ms) instead of an ISO ``timestamp`` column.
        # - ``"option_quotes"`` — same ``timestamp_ms`` column.
        # - ``"paper_trades"`` — only deletes closed trades.
        # Tables with epoch-ms timestamps need a different cutoff
        _MS_TABLES = {"option_chain_snapshots", "option_quotes"}

        for table, days in retention_days.items():
            if table in ("paper_trades",):
                continue  # handled below
            if table in _MS_TABLES:
                cutoff_ms = int(
                    (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
                )
                await self._connection.execute(
                    f"DELETE FROM {table} WHERE timestamp_ms < ?",
                    (cutoff_ms,),
                )
                logger.info("Cleaned up %s older than %d days (cutoff_ms=%d)", table, days, cutoff_ms)
            else:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                await self._connection.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?",
                    (cutoff,)
                )
                logger.info(f"Cleaned up {table} older than {days} days")

        await self._connection.commit()

        # Clean up old closed paper trades (keep open ones)
        if 'paper_trades' in retention_days:
            days = retention_days['paper_trades']
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            await self._connection.execute(
                "DELETE FROM paper_trades WHERE status != 'open' AND close_timestamp < ?",
                (cutoff,)
            )
            logger.info(f"Cleaned up closed paper_trades older than {days} days")
            await self._connection.commit()
    
    async def vacuum(self) -> None:
        # Optimize database by reclaiming space.
        await self._connection.execute("VACUUM")
        logger.info("Database vacuumed")

    async def run_maintenance(self) -> Dict[str, Any]:
        # Run periodic maintenance: PRAGMA optimize + retention cleanup.
        await self._connection.execute("PRAGMA optimize")
        logger.info("PRAGMA optimize completed")
        return await self.get_db_stats()

# Bar Query Operations (for restart dedupe)

    async def get_latest_bar_ts(
        self, source: str, symbol: str, bar_duration: int = 60
    ) -> Optional[int]:
        # Get the latest bar timestamp (ms) for a provider+symbol+timeframe.
        #
        # Used by connectors on startup to initialize last_bar_ts for dedupe.
        #
        # Parameters
        # source : str
        # Provider name (e.g., 'alpaca', 'bybit').
        # symbol : str
        # Trading symbol.
        # bar_duration : int
        # Bar duration in seconds (default 60 for 1m bars).
        #
        # Returns
        # int or None
        # Latest bar timestamp in milliseconds (bar open time), or None if no bars.
        cursor = await self._connection.execute(
            """
            SELECT timestamp FROM market_bars
            WHERE source = ? AND symbol = ? AND bar_duration = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (source, symbol, bar_duration),
        )
        row = await cursor.fetchone()
        if row and row['timestamp']:
            # Parse ISO timestamp to ms
            ts_str = row['timestamp']
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        return None

    async def get_latest_bar_close(
        self, source: str, symbol: str, bar_duration: int = 60
    ) -> Optional[float]:
        # Get the close price of the most recent bar for a source+symbol (e.g. for underlying price in IV enrichment).
        # Returns None if no bars exist.
        cursor = await self._connection.execute(
            """
            SELECT close FROM market_bars
            WHERE source = ? AND symbol = ? AND bar_duration = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (source, symbol, bar_duration),
        )
        row = await cursor.fetchone()
        if row and row['close'] is not None:
            return float(row['close'])
        return None

    async def get_db_stats(self) -> Dict[str, Any]:
        # Get database size and table row counts.
        import os
        db_size = os.path.getsize(str(self.db_path)) if self.db_path.exists() else 0
        tables = [
            'detections', 'funding_rates', 'options_iv', 'liquidations',
            'price_snapshots', 'system_health', 'daily_stats',
            'market_bars', 'signal_events', 'market_metrics',
        ]
        # Also check paper_trades if it exists
        try:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'"
            )
            if await cursor.fetchone():
                tables.append('paper_trades')
        except Exception:
            pass

        row_counts = {}
        for table in tables:
            try:
                cursor = await self._connection.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                row = await cursor.fetchone()
                row_counts[table] = row['cnt'] if row else 0
            except Exception:
                row_counts[table] = -1

        return {
            'db_size_bytes': db_size,
            'db_size_mb': round(db_size / (1024 * 1024), 1),
            'row_counts': row_counts,
        }

    def get_size_mb(self) -> float:
        # Return DB size in MB without row scans (fast).
        db_size = os.path.getsize(str(self.db_path)) if self.db_path.exists() else 0
        return round(db_size / (1024 * 1024), 1)

    async def reset_paper_epoch(self, starting_equity: float, scope: str = 'all', reason: str = 'manual_reset') -> str:
        # Start a new paper equity epoch. Old data remains but is excluded from current metrics.
        epoch_start = datetime.now(timezone.utc).isoformat()
        await self._connection.execute("""
            INSERT INTO paper_equity_epochs (epoch_start, starting_equity, reason, scope)
            VALUES (?, ?, ?, ?)
        """, (epoch_start, starting_equity, reason, scope))
        await self._connection.commit()
        logger.info(f"New paper equity epoch started: equity=${starting_equity}, scope={scope}, reason={reason}")
        return epoch_start

    async def get_current_epoch_start(self) -> Optional[str]:
        # Get the start timestamp of the current paper equity epoch.
        cursor = await self._connection.execute(
            "SELECT epoch_start FROM paper_equity_epochs ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row['epoch_start'] if row else None

# Regime Operations

    async def write_regime(
        self,
        event_type: str,
        scope: str,
        timeframe: int,
        timestamp_ms: int,
        config_hash: str,
        vol_regime: Optional[str] = None,
        trend_regime: Optional[str] = None,
        liquidity_regime: Optional[str] = None,
        session_regime: Optional[str] = None,
        risk_regime: Optional[str] = None,
        spread_pct: Optional[float] = None,
        volume_pctile: Optional[float] = None,
        confidence: float = 0.0,
        is_warm: bool = False,
        data_quality_flags: int = 0,
        metrics_json: Optional[str] = None,
    ) -> None:
        # Upsert a regime event.
        #
        # Uses INSERT OR REPLACE for idempotency.
        # All new fields (liquidity_regime, spread_pct, volume_pctile) are
        # optional and default to NULL for backward compatibility.
        await self._connection.execute("""
            INSERT OR REPLACE INTO regimes (
                event_type, scope, timeframe, timestamp, config_hash,
                vol_regime, trend_regime, liquidity_regime,
                session_regime, risk_regime,
                spread_pct, volume_pctile,
                confidence, is_warm, data_quality_flags, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_type, scope, timeframe, timestamp_ms, config_hash,
            vol_regime, trend_regime, liquidity_regime,
            session_regime, risk_regime,
            spread_pct, volume_pctile,
            confidence, 1 if is_warm else 0, data_quality_flags, metrics_json,
        ))
        await self._connection.commit()

    async def get_regimes(
        self,
        scope: str,
        *,
        event_type: Optional[str] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 100_000,
    ) -> List[Dict[str, Any]]:
        # Load regime rows for a given scope (symbol or market).
        #
        # Parameters
        # scope : str
        # Symbol (e.g. "SPY") or market (e.g. "EQUITIES").
        # event_type : str, optional
        # Filter by "symbol" or "market". If None, returns both.
        # start_ms / end_ms : int, optional
        # Epoch-ms range filter on ``timestamp``.
        # limit : int
        # Max rows returned (default 100k).
        #
        # Returns
        # List[Dict[str, Any]]
        # Rows ordered by timestamp ascending, each with ``timestamp_ms``
        # normalised from the DB ``timestamp`` column.
        clauses = ["scope = ?"]
        params: list = [scope]
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if start_ms is not None:
            clauses.append("timestamp >= ?")
            params.append(start_ms)
        if end_ms is not None:
            clauses.append("timestamp <= ?")
            params.append(end_ms)
        where = " AND ".join(clauses)
        params.append(limit)

        cursor = await self._connection.execute(
            f"SELECT * FROM regimes WHERE {where} ORDER BY timestamp ASC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            # Normalise the column name for consumers
            d["timestamp_ms"] = d.pop("timestamp", d.get("timestamp_ms", 0))
            results.append(d)
        return results

    async def get_latest_regime(
        self,
        scope: str,
        asof_ms: int,
        event_type: str = "symbol",
    ) -> Optional[Dict[str, Any]]:
        # Return the most recent regime for *scope* at or before *asof_ms*.
        #
        # This is the primary helper for replay: it answers
        # "what regime was in effect for SPY at simulation time T?"
        cursor = await self._connection.execute(
            """SELECT * FROM regimes
               WHERE scope = ? AND event_type = ? AND timestamp <= ?
               ORDER BY timestamp DESC LIMIT 1""",
            (scope, event_type, asof_ms),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cursor.description] if cursor.description else []
        d = dict(zip(cols, row))
        d["timestamp_ms"] = d.pop("timestamp", d.get("timestamp_ms", 0))
        return d

    async def get_option_chain_snapshots(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        limit: int = 100_000,
    ) -> List[Dict[str, Any]]:
        # Load option chain snapshot rows for replay.
        cursor = await self._connection.execute(
            """SELECT * FROM option_chain_snapshots
               WHERE symbol = ? AND timestamp_ms >= ? AND timestamp_ms <= ?
               ORDER BY timestamp_ms ASC LIMIT ?""",
            (symbol, start_ms, end_ms, limit),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(cols, row)) for row in rows]

    async def get_recent_bars(
        self,
        source: str,
        symbol: str,
        timeframe: int,
        n: int,
    ) -> List[Dict[str, Any]]:
        # Get the most recent N bars for warmup.
        #
        # Returns bars in ascending timestamp order (oldest first).
        cursor = await self._connection.execute("""
            SELECT * FROM market_bars
            WHERE source = ? AND symbol = ? AND bar_duration = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (source, symbol, timeframe, n))
        rows = await cursor.fetchall()
        
        # Convert to dicts and reverse to get ascending order
        bars = [dict(row) for row in rows]
        bars.reverse()
        return bars

    async def get_latest_bar_ts(
        self,
        source: str,
        symbol: str,
        bar_duration: int = 60,
    ) -> Optional[int]:
        # Get the latest bar timestamp in milliseconds.
        #
        # Returns None if no bars exist for this source/symbol/bar_duration.
        cursor = await self._connection.execute("""
            SELECT timestamp FROM market_bars
            WHERE source = ? AND symbol = ? AND bar_duration = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (source, symbol, bar_duration))
        row = await cursor.fetchone()
        if row is None:
            return None
        # timestamp is stored as ISO string, convert to ms
        from datetime import datetime
        ts_str = row['timestamp']
        try:
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

# Signal Operations (Phase 3)

    async def write_signal(
        self,
        idempotency_key: str,
        timestamp_ms: int,
        strategy_id: str,
        config_hash: str,
        symbol: str,
        direction: str,
        signal_type: str,
        timeframe: int,
        entry_type: Optional[str] = None,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        horizon: Optional[str] = None,
        confidence: float = 1.0,
        quality_score: int = 50,
        data_quality_flags: int = 0,
        regime_snapshot_json: Optional[str] = None,
        features_snapshot_json: Optional[str] = None,
        explain: str = "",
    ) -> None:
        # Upsert a signal event.
        #
        # Uses INSERT OR IGNORE for idempotency (idempotency_key is UNIQUE).
        await self._connection.execute("""
            INSERT OR IGNORE INTO signals (
                idempotency_key, timestamp_ms, strategy_id, config_hash,
                symbol, direction, signal_type, timeframe,
                entry_type, entry_price, stop_price, tp_price, horizon,
                confidence, quality_score, data_quality_flags,
                regime_snapshot_json, features_snapshot_json, explain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            idempotency_key, timestamp_ms, strategy_id, config_hash,
            symbol, direction, signal_type, timeframe,
            entry_type, entry_price, stop_price, tp_price, horizon,
            confidence, quality_score, data_quality_flags,
            regime_snapshot_json, features_snapshot_json, explain,
        ))
        await self._connection.commit()

    async def write_signal_outcome(
        self,
        idempotency_key: str,
        timestamp_ms: int,
        symbol: str,
        strategy_id: str,
        ret_1bar: Optional[float] = None,
        ret_5bar: Optional[float] = None,
        ret_10bar: Optional[float] = None,
        ret_60bar: Optional[float] = None,
        pnl_1bar: Optional[float] = None,
        pnl_5bar: Optional[float] = None,
    ) -> None:
        # Upsert a signal outcome (markout).
        #
        # Uses INSERT OR REPLACE for updating partial markouts.
        await self._connection.execute("""
            INSERT OR REPLACE INTO signal_outcomes (
                idempotency_key, timestamp_ms, symbol, strategy_id,
                ret_1bar, ret_5bar, ret_10bar, ret_60bar,
                pnl_1bar, pnl_5bar
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            idempotency_key, timestamp_ms, symbol, strategy_id,
            ret_1bar, ret_5bar, ret_10bar, ret_60bar,
            pnl_1bar, pnl_5bar,
        ))
        await self._connection.commit()

    async def get_pending_markouts(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        # Get signals that need markout computation.
        #
        # Returns signals without corresponding outcomes.
        if strategy_id:
            cursor = await self._connection.execute("""
                SELECT s.* FROM signals s
                LEFT JOIN signal_outcomes o ON s.idempotency_key = o.idempotency_key
                WHERE o.idempotency_key IS NULL
                AND s.strategy_id = ?
                ORDER BY s.timestamp_ms DESC
                LIMIT ?
            """, (strategy_id, limit))
        else:
            cursor = await self._connection.execute("""
                SELECT s.* FROM signals s
                LEFT JOIN signal_outcomes o ON s.idempotency_key = o.idempotency_key
                WHERE o.idempotency_key IS NULL
                ORDER BY s.timestamp_ms DESC
                LIMIT ?
            """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def backup(self, backup_path: str) -> None:
        # Create a database backup.
        backup_db = Path(backup_path)
        backup_db.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(str(backup_db)) as backup:
            await self._connection.backup(backup)
        
        logger.info(f"Database backed up to {backup_path}")

# Phase 3B: Options Data Operations

    async def upsert_option_contract(
        self,
        contract_id: str,
        symbol: str,
        option_symbol: str,
        strike: float,
        expiration_ms: int,
        option_type: str,
        provider: str,
        timestamp_ms: int,
        multiplier: int = 100,
        style: str = "american",
    ) -> None:
        # Upsert an option contract (insert or update last_updated_ms).
        await self._connection.execute("""
            INSERT INTO option_contracts (
                contract_id, symbol, option_symbol, strike, expiration_ms,
                option_type, multiplier, style, provider, first_seen_ms, last_updated_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(contract_id) DO UPDATE SET
                last_updated_ms = excluded.last_updated_ms
        """, (
            contract_id, symbol, option_symbol, strike, expiration_ms,
            option_type, multiplier, style, provider, timestamp_ms, timestamp_ms
        ))
        await self._connection.commit()

    async def insert_option_quote(
        self,
        contract_id: str,
        symbol: str,
        strike: float,
        expiration_ms: int,
        option_type: str,
        bid: float,
        ask: float,
        timestamp_ms: int,
        recv_ts_ms: int | None = None,
        last: float = 0.0,
        mid: float = 0.0,
        volume: int = 0,
        open_interest: int = 0,
        iv: float = None,
        delta: float = None,
        gamma: float = None,
        theta: float = None,
        vega: float = None,
        source_ts_ms: int = None,
        provider: str = "",
    ) -> None:
        # Insert an option quote (append-only).
        await self._connection.execute("""
            INSERT INTO option_quotes (
                contract_id, symbol, strike, expiration_ms, option_type,
                bid, ask, last, mid, volume, open_interest,
                iv, delta, gamma, theta, vega,
                timestamp_ms, source_ts_ms, recv_ts_ms, provider
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contract_id, symbol, strike, expiration_ms, option_type,
            bid, ask, last, mid, volume, open_interest,
            iv, delta, gamma, theta, vega,
            timestamp_ms, source_ts_ms, recv_ts_ms, provider
        ))
        await self._connection.commit()

    async def insert_option_chain_snapshot(
        self,
        snapshot_id: str,
        symbol: str,
        expiration_ms: int,
        underlying_price: float,
        timestamp_ms: int,
        quotes_json: str,
        n_strikes: int = 0,
        atm_iv: float = None,
        source_ts_ms: int = None,
        provider: str = "",
    ) -> None:
        # Insert or replace an option chain snapshot (atomic).
        await self._connection.execute("""
            INSERT OR REPLACE INTO option_chain_snapshots (
                snapshot_id, symbol, expiration_ms, underlying_price,
                n_strikes, atm_iv, timestamp_ms, source_ts_ms, provider, quotes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_id, symbol, expiration_ms, underlying_price,
            n_strikes, atm_iv, timestamp_ms, source_ts_ms, provider, quotes_json
        ))
        await self._connection.commit()

    async def get_option_chain_snapshot(
        self,
        symbol: str,
        expiration_ms: int,
        timestamp_ms: int = None,
    ) -> Optional[Dict[str, Any]]:
        # Get an option chain snapshot.
        #
        # If timestamp_ms is None, returns the most recent snapshot.
        if timestamp_ms:
            cursor = await self._connection.execute("""
                SELECT * FROM option_chain_snapshots
                WHERE symbol = ? AND expiration_ms = ? AND timestamp_ms = ?
            """, (symbol, expiration_ms, timestamp_ms))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM option_chain_snapshots
                WHERE symbol = ? AND expiration_ms = ?
                ORDER BY timestamp_ms DESC
                LIMIT 1
            """, (symbol, expiration_ms))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_option_contracts(
        self,
        symbol: str,
        expiration_ms: int = None,
    ) -> List[Dict[str, Any]]:
        # Get option contracts for a symbol, optionally filtered by expiration.
        if expiration_ms:
            cursor = await self._connection.execute("""
                SELECT * FROM option_contracts
                WHERE symbol = ? AND expiration_ms = ?
                ORDER BY strike ASC
            """, (symbol, expiration_ms))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM option_contracts
                WHERE symbol = ?
                ORDER BY expiration_ms ASC, strike ASC
            """, (symbol,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

# Phase 4A.1: Bar Outcomes (Backtest Ground Truth)

    async def upsert_bar_outcome(
        self,
        provider: str,
        symbol: str,
        bar_duration_seconds: int,
        timestamp_ms: int,
        horizon_seconds: int,
        outcome_version: str,
        close_now: float,
        close_at_horizon: float | None,
        fwd_return: float | None,
        max_runup: float | None,
        max_drawdown: float | None,
        realized_vol: float | None,
        max_high_in_window: float | None,
        min_low_in_window: float | None,
        max_runup_ts_ms: int | None,
        max_drawdown_ts_ms: int | None,
        time_to_max_runup_ms: int | None,
        time_to_max_drawdown_ms: int | None,
        status: str,
        close_ref_ms: int,
        window_start_ms: int,
        window_end_ms: int,
        bars_expected: int,
        bars_found: int,
        gap_count: int,
        computed_at_ms: int | None,
    ) -> bool:
        # Upsert a bar outcome record (idempotent).
        #
        # Status upgrade rules:
        # - INCOMPLETE records can be updated to OK or GAP
        # - OK/GAP records are only updated if outcome_version changes
        #
        # Returns:
        # True if inserted/updated, False on error.
        try:
            await self._connection.execute("""
                INSERT INTO bar_outcomes (
                    provider, symbol, bar_duration_seconds, timestamp_ms,
                    horizon_seconds, outcome_version,
                    close_now, close_at_horizon, fwd_return, max_runup, max_drawdown,
                    realized_vol, max_high_in_window, min_low_in_window,
                    max_runup_ts_ms, max_drawdown_ts_ms,
                    time_to_max_runup_ms, time_to_max_drawdown_ms,
                    status, close_ref_ms, window_start_ms, window_end_ms,
                    bars_expected, bars_found, gap_count, computed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, symbol, bar_duration_seconds, timestamp_ms, horizon_seconds, outcome_version) DO UPDATE SET
                    close_at_horizon = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.close_at_horizon
                        ELSE bar_outcomes.close_at_horizon
                    END,
                    fwd_return = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.fwd_return
                        ELSE bar_outcomes.fwd_return
                    END,
                    max_runup = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_runup
                        ELSE bar_outcomes.max_runup
                    END,
                    max_drawdown = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_drawdown
                        ELSE bar_outcomes.max_drawdown
                    END,
                    realized_vol = excluded.realized_vol,
                    max_high_in_window = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_high_in_window
                        ELSE bar_outcomes.max_high_in_window
                    END,
                    min_low_in_window = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.min_low_in_window
                        ELSE bar_outcomes.min_low_in_window
                    END,
                    max_runup_ts_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_runup_ts_ms
                        ELSE bar_outcomes.max_runup_ts_ms
                    END,
                    max_drawdown_ts_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_drawdown_ts_ms
                        ELSE bar_outcomes.max_drawdown_ts_ms
                    END,
                    time_to_max_runup_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.time_to_max_runup_ms
                        ELSE bar_outcomes.time_to_max_runup_ms
                    END,
                    time_to_max_drawdown_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.time_to_max_drawdown_ms
                        ELSE bar_outcomes.time_to_max_drawdown_ms
                    END,
                    status = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.status
                        ELSE bar_outcomes.status
                    END,
                    bars_found = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.bars_found
                        ELSE bar_outcomes.bars_found
                    END,
                    gap_count = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.gap_count
                        ELSE bar_outcomes.gap_count
                    END,
                    computed_at_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.computed_at_ms
                        ELSE bar_outcomes.computed_at_ms
                    END
            """, (
                provider, symbol, bar_duration_seconds, timestamp_ms,
                horizon_seconds, outcome_version,
                close_now, close_at_horizon, fwd_return, max_runup, max_drawdown,
                realized_vol, max_high_in_window, min_low_in_window,
                max_runup_ts_ms, max_drawdown_ts_ms,
                time_to_max_runup_ms, time_to_max_drawdown_ms,
                status, close_ref_ms, window_start_ms, window_end_ms,
                bars_expected, bars_found, gap_count, computed_at_ms,
            ))
            await self._connection.commit()
            return True
        except Exception as e:
            logger.warning("upsert_bar_outcome failed: %s", e)
            return False

    async def upsert_bar_outcomes_batch(
        self,
        outcomes: List[tuple],
    ) -> int:
        # Batch upsert bar outcomes. Returns count of rows affected.
        #
        # Each outcome tuple should have 26 elements matching upsert_bar_outcome params.
        # Uses executemany for efficiency.
        if not outcomes:
            return 0
        try:
            await self._connection.executemany("""
                INSERT INTO bar_outcomes (
                    provider, symbol, bar_duration_seconds, timestamp_ms,
                    horizon_seconds, outcome_version,
                    close_now, close_at_horizon, fwd_return, max_runup, max_drawdown,
                    realized_vol, max_high_in_window, min_low_in_window,
                    max_runup_ts_ms, max_drawdown_ts_ms,
                    time_to_max_runup_ms, time_to_max_drawdown_ms,
                    status, close_ref_ms, window_start_ms, window_end_ms,
                    bars_expected, bars_found, gap_count, computed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, symbol, bar_duration_seconds, timestamp_ms, horizon_seconds, outcome_version) DO UPDATE SET
                    close_at_horizon = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.close_at_horizon
                        ELSE bar_outcomes.close_at_horizon
                    END,
                    fwd_return = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.fwd_return
                        ELSE bar_outcomes.fwd_return
                    END,
                    max_runup = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_runup
                        ELSE bar_outcomes.max_runup
                    END,
                    max_drawdown = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_drawdown
                        ELSE bar_outcomes.max_drawdown
                    END,
                    realized_vol = excluded.realized_vol,
                    max_high_in_window = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_high_in_window
                        ELSE bar_outcomes.max_high_in_window
                    END,
                    min_low_in_window = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.min_low_in_window
                        ELSE bar_outcomes.min_low_in_window
                    END,
                    max_runup_ts_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_runup_ts_ms
                        ELSE bar_outcomes.max_runup_ts_ms
                    END,
                    max_drawdown_ts_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.max_drawdown_ts_ms
                        ELSE bar_outcomes.max_drawdown_ts_ms
                    END,
                    time_to_max_runup_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.time_to_max_runup_ms
                        ELSE bar_outcomes.time_to_max_runup_ms
                    END,
                    time_to_max_drawdown_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.time_to_max_drawdown_ms
                        ELSE bar_outcomes.time_to_max_drawdown_ms
                    END,
                    status = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.status
                        ELSE bar_outcomes.status
                    END,
                    bars_found = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.bars_found
                        ELSE bar_outcomes.bars_found
                    END,
                    gap_count = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.gap_count
                        ELSE bar_outcomes.gap_count
                    END,
                    computed_at_ms = CASE
                        WHEN bar_outcomes.status = 'INCOMPLETE' THEN excluded.computed_at_ms
                        ELSE bar_outcomes.computed_at_ms
                    END
            """, outcomes)
            await self._connection.commit()
            return len(outcomes)
        except Exception as e:
            logger.warning("upsert_bar_outcomes_batch failed: %s", e)
            return 0

    async def get_bars_for_outcome_computation(
        self,
        source: str,
        symbol: str,
        bar_duration: int,
        start_ms: int,
        end_ms: int,
        limit: int = 100000,
    ) -> List[Dict[str, Any]]:
        # Fetch bars for outcome computation, ordered by timestamp.
        #
        # Returns bars in [start_ms, end_ms] range plus lookahead bars
        # needed for forward returns.
        cursor = await self._connection.execute("""
            SELECT 
                source, symbol, bar_duration, timestamp,
                open, high, low, close, volume
            FROM market_bars
            WHERE source = ? AND symbol = ? AND bar_duration = ?
              AND timestamp >= datetime(?, 'unixepoch')
            ORDER BY timestamp ASC
            LIMIT ?
        """, (source, symbol, bar_duration, start_ms / 1000.0, limit))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_bar_outcomes(
        self,
        provider: str,
        symbol: str,
        bar_duration_seconds: int | None = None,
        horizon_seconds: int | None = None,
        status: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        # Query bar outcomes with optional filters.
        sql = "SELECT * FROM bar_outcomes WHERE provider = ? AND symbol = ?"
        params: list = [provider, symbol]
        
        if bar_duration_seconds is not None:
            sql += " AND bar_duration_seconds = ?"
            params.append(bar_duration_seconds)
        if horizon_seconds is not None:
            sql += " AND horizon_seconds = ?"
            params.append(horizon_seconds)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if start_ms is not None:
            sql += " AND timestamp_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND timestamp_ms <= ?"
            params.append(end_ms)
        
        sql += " ORDER BY timestamp_ms ASC LIMIT ?"
        params.append(limit)
        
        cursor = await self._connection.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_outcome_coverage_stats(
        self,
        provider: str | None = None,
        symbol: str | None = None,
    ) -> Dict[str, Any]:
        # Get coverage statistics for bar outcomes.
        #
        # Returns min/max timestamps, counts by status, and coverage metrics.
        where_parts = []
        params: list = []
        if provider:
            where_parts.append("provider = ?")
            params.append(provider)
        if symbol:
            where_parts.append("symbol = ?")
            params.append(symbol)
        
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        
        cursor = await self._connection.execute(f"""
            SELECT 
                COUNT(*) as total_outcomes,
                MIN(timestamp_ms) as min_ts_ms,
                MAX(timestamp_ms) as max_ts_ms,
                SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) as ok_count,
                SUM(CASE WHEN status = 'INCOMPLETE' THEN 1 ELSE 0 END) as incomplete_count,
                SUM(CASE WHEN status = 'GAP' THEN 1 ELSE 0 END) as gap_count
            FROM bar_outcomes
            {where_clause}
        """, tuple(params))
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def get_bar_coverage_stats(
        self,
        source: str | None = None,
        symbol: str | None = None,
    ) -> Dict[str, Any]:
        # Get coverage statistics for market_bars.
        #
        # Returns min/max timestamps, total count, and span.
        where_parts = []
        params: list = []
        if source:
            where_parts.append("source = ?")
            params.append(source)
        if symbol:
            where_parts.append("symbol = ?")
            params.append(symbol)
        
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        
        cursor = await self._connection.execute(f"""
            SELECT 
                COUNT(*) as total_bars,
                MIN(timestamp) as min_ts,
                MAX(timestamp) as max_ts
            FROM market_bars
            {where_clause}
        """, tuple(params))
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def get_bar_inventory(self) -> List[Dict[str, Any]]:
        # List all distinct (source, symbol, bar_duration) keys in market_bars.
        #
        # Returns rows with: source, symbol, bar_duration, count, min_ts, max_ts.
        # Used by the CLI ``list`` command so users can discover exact key strings
        # without needing a sqlite3 shell.
        cursor = await self._connection.execute("""
            SELECT
                source,
                symbol,
                bar_duration,
                COUNT(*) as bar_count,
                MIN(timestamp) as min_ts,
                MAX(timestamp) as max_ts
            FROM market_bars
            GROUP BY source, symbol, bar_duration
            ORDER BY source, symbol, bar_duration
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_outcome_inventory(self) -> List[Dict[str, Any]]:
        # List all distinct (provider, symbol, bar_duration_seconds, horizon_seconds)
        # keys in bar_outcomes with counts by status.
        #
        # Returns rows with: provider, symbol, bar_duration_seconds, horizon_seconds,
        # total, ok_count, incomplete_count, gap_count, min_ts_ms, max_ts_ms.
        cursor = await self._connection.execute("""
            SELECT
                provider,
                symbol,
                bar_duration_seconds,
                horizon_seconds,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) as ok_count,
                SUM(CASE WHEN status = 'INCOMPLETE' THEN 1 ELSE 0 END) as incomplete_count,
                SUM(CASE WHEN status = 'GAP' THEN 1 ELSE 0 END) as gap_count,
                MIN(timestamp_ms) as min_ts_ms,
                MAX(timestamp_ms) as max_ts_ms
            FROM bar_outcomes
            GROUP BY provider, symbol, bar_duration_seconds, horizon_seconds
            ORDER BY provider, symbol, bar_duration_seconds, horizon_seconds
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

# System Heartbeat (uptime tracking)

    async def write_heartbeat(
        self,
        component: str,
        timestamp_ms: int,
        metadata_json: Optional[str] = None,
    ) -> None:
        # Write a system heartbeat (idempotent via PK).
        await self._connection.execute(
            "INSERT OR IGNORE INTO system_heartbeat "
            "(component, timestamp_ms, metadata_json) VALUES (?, ?, ?)",
            (component, timestamp_ms, metadata_json),
        )
        await self._connection.commit()

    async def get_heartbeats(
        self,
        component: str,
        start_ms: int,
        end_ms: int,
    ) -> List[int]:
        # Return sorted list of heartbeat timestamp_ms for a component in range.
        cursor = await self._connection.execute(
            "SELECT timestamp_ms FROM system_heartbeat "
            "WHERE component = ? AND timestamp_ms >= ? AND timestamp_ms <= ? "
            "ORDER BY timestamp_ms ASC",
            (component, start_ms, end_ms),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_bar_timestamps(
        self,
        source: str,
        symbol: str,
        bar_duration: int,
        start_ms: int,
        end_ms: int,
    ) -> List[int]:
        # Return sorted list of bar open timestamps (as epoch ms) for gap analysis.
        #
        # Converts the ISO-string ``timestamp`` column to epoch ms using
        # ``strftime`` in SQLite for efficiency (avoids pulling all rows into Python).
        cursor = await self._connection.execute(
            "SELECT CAST(strftime('%s', timestamp) AS INTEGER) * 1000 as ts_ms "
            "FROM market_bars "
            "WHERE source = ? AND symbol = ? AND bar_duration = ? "
            "  AND timestamp >= datetime(?, 'unixepoch') "
            "  AND timestamp <= datetime(?, 'unixepoch') "
            "ORDER BY timestamp ASC",
            (source, symbol, bar_duration,
             start_ms / 1000.0, end_ms / 1000.0),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_heartbeat_inventory(self) -> List[Dict[str, Any]]:
        # List distinct components in system_heartbeat with counts and ranges.
        #
        # Returns rows with: component, count, min_ts_ms, max_ts_ms.
        cursor = await self._connection.execute("""
            SELECT
                component,
                COUNT(*) as count,
                MIN(timestamp_ms) as min_ts_ms,
                MAX(timestamp_ms) as max_ts_ms
            FROM system_heartbeat
            GROUP BY component
            ORDER BY component
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

# Bar backfill helpers

    async def get_last_bar_timestamp_ms(
        self,
        source: str,
        symbol: str,
        bar_duration: int = 60,
    ) -> Optional[int]:
        # Return the latest bar timestamp as epoch ms, or None if no bars.
        cursor = await self._connection.execute(
            "SELECT MAX(timestamp) FROM market_bars "
            "WHERE source = ? AND symbol = ? AND bar_duration = ?",
            (source, symbol, bar_duration),
        )
        row = await cursor.fetchone()
        if row and row[0]:
            try:
                from datetime import datetime as _dt, timezone as _tz
                ts_str = row[0]
                if "T" in ts_str:
                    dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    dt = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None
        return None

    async def get_first_bar_timestamp_ms(
        self,
        source: str,
        symbol: str,
        bar_duration: int = 60,
    ) -> Optional[int]:
        # Return the earliest bar timestamp as epoch ms, or None if no bars.
        cursor = await self._connection.execute(
            "SELECT MIN(timestamp) FROM market_bars "
            "WHERE source = ? AND symbol = ? AND bar_duration = ?",
            (source, symbol, bar_duration),
        )
        row = await cursor.fetchone()
        if row and row[0]:
            try:
                from datetime import datetime as _dt, timezone as _tz
                ts_str = row[0]
                if "T" in ts_str:
                    dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    dt = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None
        return None

    async def upsert_bars_backfill(self, rows: List[tuple]) -> int:
        # Insert backfilled bars using INSERT OR IGNORE.
        #
        # Existing bars (by unique key) are NOT overwritten.
        # This ensures live-collected bars (which have tick-level fidelity)
        # are never replaced by REST backfill bars.
        #
        # Parameters
        # rows : list of tuple
        # Each tuple matches market_bars columns:
        # (timestamp, symbol, source, open, high, low, close, volume,
        # tick_count, n_ticks, first_source_ts, last_source_ts,
        # late_ticks_dropped, close_reason, bar_duration)
        #
        # Returns
        # int
        # Number of rows actually inserted (new bars).
        if not rows:
            return 0

        cursor = await self._connection.execute("SELECT COUNT(*) FROM market_bars")
        before = (await cursor.fetchone())[0]

        await self._connection.executemany(
            """INSERT OR IGNORE INTO market_bars
               (timestamp, symbol, source, open, high, low, close, volume,
                tick_count, n_ticks, first_source_ts, last_source_ts,
                late_ticks_dropped, close_reason, bar_duration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self._connection.commit()

        cursor = await self._connection.execute("SELECT COUNT(*) FROM market_bars")
        after = (await cursor.fetchone())[0]
        return after - before

    async def get_bar_health(self) -> List[Dict[str, Any]]:
        # Return last-bar timestamps per source/symbol for health reporting.
        #
        # Returns rows with: source, symbol, bar_duration, bar_count,
        # last_ts (text), last_ts_age_s (seconds since last bar, approximate).
        cursor = await self._connection.execute("""
            SELECT
                source,
                symbol,
                bar_duration,
                COUNT(*) as bar_count,
                MAX(timestamp) as last_ts
            FROM market_bars
            GROUP BY source, symbol, bar_duration
            ORDER BY source, symbol
        """)
        rows = await cursor.fetchall()
        result = []
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        for row in rows:
            d = dict(row)
            try:
                ts_str = d["last_ts"]
                if "T" in ts_str:
                    dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    dt = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                d["last_ts_age_s"] = int((now - dt).total_seconds())
            except Exception:
                d["last_ts_age_s"] = -1
            result.append(d)
        return result

    async def get_bars_daily_for_risk_flow(
        self,
        source: str,
        symbols: List[str],
        end_ms: int,
        lookback_days: int = 365,
    ) -> Dict[str, List[Dict[str, Any]]]:
        # Fetch daily bars for risk-flow computation.
        #
        # Parameters
        # source : str
        # Bar source (e.g. ``"alphavantage"``).
        # symbols : list[str]
        # Symbols to retrieve (e.g. ``["SPY", "EWJ", "FX:EURUSD"]``).
        # end_ms : int
        # Strict upper bound (exclusive) — bars with timestamp < end_ms.
        # lookback_days : int
        # How far back to look (default 365).
        #
        # Returns
        # dict[str, list[dict]]
        # ``{symbol: [{timestamp_ms, open, high, low, close, volume}, ...]}``
        # Bars are sorted ascending by timestamp.
        start_epoch = (end_ms / 1000.0) - (lookback_days * 86400)
        result: Dict[str, List[Dict[str, Any]]] = {}
        for sym in symbols:
            cursor = await self._connection.execute("""
                SELECT
                    CAST(strftime('%s', timestamp) AS INTEGER) * 1000 as timestamp_ms,
                    open, high, low, close, volume
                FROM market_bars
                WHERE source = ? AND symbol = ? AND bar_duration = 86400
                  AND timestamp >= datetime(?, 'unixepoch')
                  AND timestamp < datetime(?, 'unixepoch')
                ORDER BY timestamp ASC
            """, (source, sym, start_epoch, end_ms / 1000.0))
            rows = await cursor.fetchall()
            result[sym] = [dict(row) for row in rows]
        return result
