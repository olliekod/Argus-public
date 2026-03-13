"""
Tests for PnL computation, zombie detection, uniformity monitor, and best-trader scoring.

Usage: python -m pytest tests/test_pnl_and_zombies.py -v
"""

import asyncio
import os
import sys
import tempfile
import math
from datetime import datetime, timedelta, timezone

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import modules directly to avoid pulling in the full orchestrator tree.
# We register stub parent packages so relative imports resolve without
# triggering the real src/__init__.py which imports the entire app.
import importlib.util
import types

def _setup_stub_packages():
    """Register minimal stub packages so relative imports work."""
    for pkg_name in ('src', 'src.core', 'src.analysis'):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [os.path.join(project_root, *pkg_name.split('.'))]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

_setup_stub_packages()

def _import_module(name, file_path):
    """Import a module by file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load logger first (dependency of uniformity_monitor)
_logger_mod = _import_module(
    'src.core.logger',
    os.path.join(project_root, 'src', 'core', 'logger.py')
)

# Load database
_db_mod = _import_module(
    'src.core.database',
    os.path.join(project_root, 'src', 'core', 'database.py')
)
Database = _db_mod.Database

# Load uniformity monitor
_um_mod = _import_module(
    'src.analysis.uniformity_monitor',
    os.path.join(project_root, 'src', 'analysis', 'uniformity_monitor.py')
)
compute_hhi = _um_mod.compute_hhi
compute_entropy = _um_mod.compute_entropy
analyze_variable = _um_mod.analyze_variable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


async def make_db():
    """Create a fresh in-memory database for testing.

    Calls _create_tables to set up the full schema including paper_trades.
    """
    db = Database(":memory:")
    await db.connect()
    # _create_tables is called by connect(), but the paper_trades table is
    # created in PaperTraderFarm.initialize(). We must create it manually here.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id TEXT PRIMARY KEY,
            trader_id TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            strikes TEXT,
            expiry TEXT,
            entry_credit REAL,
            contracts INTEGER DEFAULT 1,
            status TEXT DEFAULT 'open',
            close_timestamp TEXT,
            close_price REAL,
            realized_pnl REAL,
            market_conditions TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_trader ON paper_trades(trader_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)"
    )
    await db._connection.commit()
    return db


async def insert_trade(db, trader_id, strategy_type="bull_put", symbol="IBIT",
                       pnl=10.0, status="closed", days_ago=1,
                       expiry=None, strikes="55/53"):
    """Insert a paper trade for testing."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    close_ts = datetime.now(timezone.utc).isoformat() if status != 'open' else None
    trade_id = f"test-{trader_id}-{days_ago}-{pnl}"
    expiry = expiry or (datetime.now(timezone.utc) + timedelta(days=30)).strftime('%Y-%m-%d')
    await db.execute("""
        INSERT INTO paper_trades
        (id, trader_id, strategy_type, symbol, timestamp, strikes, expiry,
         entry_credit, contracts, status, close_timestamp, realized_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trade_id, trader_id, strategy_type, symbol, ts, strikes,
          expiry, 0.50, 1, status, close_ts, pnl if status != 'open' else None))
    await db._connection.commit()
    return trade_id


# ---------------------------------------------------------------------------
# PnL Tests
# ---------------------------------------------------------------------------

class TestPerTraderPnl:
    """Test per-trader PnL computation correctness."""

    def test_single_trader_return(self):
        """Return = total_pnl / starting_balance * 100."""
        async def _test():
            db = await make_db()
            # Trader with $50 profit on $5000 base = 1% return
            await insert_trade(db, "PT-000001", pnl=25.0, days_ago=5)
            await insert_trade(db, "PT-000001", pnl=25.0, days_ago=3)

            rows = await db.get_per_trader_pnl(days=30, min_trades=1)
            assert len(rows) == 1
            row = rows[0]
            assert row['trader_id'] == "PT-000001"
            assert row['total_pnl'] == 50.0
            assert row['closed_trades'] == 2
            assert row['wins'] == 2
            await db.close()

        run(_test())

    def test_multiple_traders_independent(self):
        """Each trader's return is independent — no cross-contamination."""
        async def _test():
            db = await make_db()
            await insert_trade(db, "PT-000001", pnl=100.0, days_ago=2)
            await insert_trade(db, "PT-000002", pnl=-50.0, days_ago=2)
            await insert_trade(db, "PT-000003", pnl=10.0, days_ago=2)

            rows = await db.get_per_trader_pnl(days=30, min_trades=1)
            by_id = {r['trader_id']: r for r in rows}

            assert by_id['PT-000001']['total_pnl'] == 100.0
            assert by_id['PT-000002']['total_pnl'] == -50.0
            assert by_id['PT-000003']['total_pnl'] == 10.0
            await db.close()

        run(_test())

    def test_min_trades_filter(self):
        """Traders below min_trades threshold are excluded."""
        async def _test():
            db = await make_db()
            # Trader 1: 5 trades
            for i in range(5):
                await insert_trade(db, "PT-000001", pnl=10.0, days_ago=i + 1)
            # Trader 2: 1 trade
            await insert_trade(db, "PT-000002", pnl=100.0, days_ago=1)

            rows = await db.get_per_trader_pnl(days=30, min_trades=3)
            assert len(rows) == 1
            assert rows[0]['trader_id'] == "PT-000001"
            await db.close()

        run(_test())

    def test_window_filter(self):
        """Only trades within the window are counted."""
        async def _test():
            db = await make_db()
            # Recent trade
            await insert_trade(db, "PT-000001", pnl=50.0, days_ago=2)
            # Old trade (beyond 7-day window)
            await insert_trade(db, "PT-000001", pnl=500.0, days_ago=15)

            rows = await db.get_per_trader_pnl(days=7, min_trades=1)
            assert len(rows) == 1
            # Should only include the recent $50 trade
            assert rows[0]['total_pnl'] == 50.0
            await db.close()

        run(_test())

    def test_open_positions_excluded_from_realized(self):
        """Open positions should not count towards realized PnL."""
        async def _test():
            db = await make_db()
            await insert_trade(db, "PT-000001", pnl=50.0, days_ago=2, status="closed")
            await insert_trade(db, "PT-000001", pnl=0, days_ago=1, status="open")

            rows = await db.get_per_trader_pnl(days=30, min_trades=1)
            assert len(rows) == 1
            assert rows[0]['closed_trades'] == 1  # Only the closed trade
            assert rows[0]['total_pnl'] == 50.0  # Only realized
            await db.close()

        run(_test())


# ---------------------------------------------------------------------------
# Zombie Detection Tests
# ---------------------------------------------------------------------------

class TestZombieDetection:
    """Test zombie position detection rules."""

    def test_stale_open_position(self):
        """Positions open for > stale_hours should be detected as zombies."""
        async def _test():
            db = await make_db()
            # Open position from 72 hours ago (> 48h threshold)
            await insert_trade(db, "PT-000001", pnl=0, days_ago=3, status="open",
                             expiry=(datetime.now(timezone.utc) + timedelta(days=30)).strftime('%Y-%m-%d'))

            zombies = await db.get_zombie_positions(stale_hours=48)
            assert len(zombies) == 1
            assert zombies[0]['trader_id'] == "PT-000001"
            await db.close()

        run(_test())

    def test_past_expiry_open_position(self):
        """Open positions with past expiry should be detected as zombies."""
        async def _test():
            db = await make_db()
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
            await insert_trade(db, "PT-000001", pnl=0, days_ago=1, status="open",
                             expiry=yesterday)

            zombies = await db.get_zombie_positions(stale_hours=48)
            assert len(zombies) == 1
            await db.close()

        run(_test())

    def test_fresh_open_position_not_zombie(self):
        """Recent open positions with future expiry should NOT be zombies."""
        async def _test():
            db = await make_db()
            future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime('%Y-%m-%d')
            # Inserted "0 days ago" = now
            ts = datetime.now(timezone.utc).isoformat()
            await db.execute("""
                INSERT INTO paper_trades
                (id, trader_id, strategy_type, symbol, timestamp, strikes, expiry,
                 entry_credit, contracts, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("fresh-1", "PT-000001", "bull_put", "IBIT", ts, "55/53",
                  future, 0.50, 1, "open"))
            await db._connection.commit()

            zombies = await db.get_zombie_positions(stale_hours=48)
            assert len(zombies) == 0
            await db.close()

        run(_test())

    def test_mark_zombies(self):
        """mark_zombies should update status and add reason."""
        async def _test():
            db = await make_db()
            trade_id = await insert_trade(db, "PT-000001", pnl=0, days_ago=5, status="open",
                                        expiry=(datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d'))

            count = await db.mark_zombies([trade_id], reason='test_zombie')
            assert count == 1

            # Verify status changed
            row = await db.fetch_one(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,)
            )
            assert row['status'] == 'expired'
            await db.close()

        run(_test())


# ---------------------------------------------------------------------------
# Uniformity Monitor Tests
# ---------------------------------------------------------------------------

class TestUniformityMonitor:
    """Test HHI, entropy, and convergence detection."""

    def test_hhi_all_same(self):
        """HHI = 1.0 when all values are identical."""
        values = ['A'] * 100
        assert compute_hhi(values) == 1.0

    def test_hhi_perfectly_uniform(self):
        """HHI = 1/N when values are perfectly uniform."""
        values = ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25
        hhi = compute_hhi(values)
        assert abs(hhi - 0.25) < 0.001  # 1/4 = 0.25

    def test_entropy_all_same(self):
        """Entropy = 0 when all values are identical."""
        values = ['A'] * 100
        assert compute_entropy(values) == 0.0

    def test_entropy_uniform(self):
        """Entropy = log2(N) when values are perfectly uniform."""
        values = ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25
        entropy = compute_entropy(values)
        assert abs(entropy - 2.0) < 0.01  # log2(4) = 2.0

    def test_analyze_triggers_alert_on_convergence(self):
        """analyze_variable should alert when one value dominates."""
        # 95% same value = clear bug signal
        values = ['$55/$53'] * 95 + ['$57/$55'] * 5
        result = analyze_variable(values, 'strikes')
        assert result['is_alert'] is True
        assert result['modal_pct'] > 0.9

    def test_analyze_no_alert_on_diversity(self):
        """analyze_variable should not alert when choices are diverse."""
        # 10 different values, roughly equal
        values = [f"strike_{i}" for i in range(10)] * 10
        result = analyze_variable(values, 'strikes')
        assert result['is_alert'] is False

    def test_analyze_skips_small_samples(self):
        """Should skip analysis when sample is too small."""
        values = ['A'] * 5
        result = analyze_variable(values, 'test')
        assert result.get('skipped') is True


# ---------------------------------------------------------------------------
# Best Trader Scoring Tests
# ---------------------------------------------------------------------------

class TestBestTraderScoring:
    """Test the scoring function for best trader selection."""

    def test_score_positive_pnl_wins(self):
        """Trader with positive PnL should score higher than negative."""
        from scripts.select_best_trader import score_trader

        winner = {
            'total_pnl': 200.0, 'closed_trades': 10, 'wins': 7,
            'avg_pnl': 20.0, 'worst_trade': -15.0,
        }
        loser = {
            'total_pnl': -100.0, 'closed_trades': 10, 'wins': 3,
            'avg_pnl': -10.0, 'worst_trade': -80.0,
        }
        assert score_trader(winner) > score_trader(loser)

    def test_score_zero_trades(self):
        """Trader with zero trades should get -inf score."""
        from scripts.select_best_trader import score_trader
        row = {'total_pnl': 0, 'closed_trades': 0, 'wins': 0, 'avg_pnl': 0, 'worst_trade': 0}
        assert score_trader(row) == float('-inf')

    def test_drawdown_penalty(self):
        """Trader with large single loss should score lower."""
        from scripts.select_best_trader import score_trader

        safe = {
            'total_pnl': 100.0, 'closed_trades': 10, 'wins': 7,
            'avg_pnl': 10.0, 'worst_trade': -10.0,
        }
        risky = {
            'total_pnl': 100.0, 'closed_trades': 10, 'wins': 7,
            'avg_pnl': 10.0, 'worst_trade': -500.0,
        }
        assert score_trader(safe) > score_trader(risky)


# ---------------------------------------------------------------------------
# Followed Traders DB Tests
# ---------------------------------------------------------------------------

class TestFollowedTraders:
    """Test followed traders DB operations."""

    def test_set_and_get(self):
        async def _test():
            db = await make_db()
            traders = [
                {'trader_id': 'PT-000001', 'score': 1.5, 'scoring_method': 'test',
                 'window_days': 30, 'config_json': '{}'},
                {'trader_id': 'PT-000002', 'score': 1.2, 'scoring_method': 'test',
                 'window_days': 30, 'config_json': '{}'},
            ]
            await db.set_followed_traders(traders)
            result = await db.get_followed_traders()
            assert len(result) == 2
            # Should be sorted by score desc
            assert result[0]['trader_id'] == 'PT-000001'
            assert result[1]['trader_id'] == 'PT-000002'
            await db.close()

        run(_test())

    def test_replace_on_set(self):
        """set_followed_traders should replace, not append."""
        async def _test():
            db = await make_db()
            await db.set_followed_traders([
                {'trader_id': 'PT-000001', 'score': 1.0, 'scoring_method': 'v1',
                 'window_days': 30, 'config_json': '{}'},
            ])
            await db.set_followed_traders([
                {'trader_id': 'PT-000099', 'score': 2.0, 'scoring_method': 'v2',
                 'window_days': 60, 'config_json': '{}'},
            ])
            result = await db.get_followed_traders()
            assert len(result) == 1
            assert result[0]['trader_id'] == 'PT-000099'
            await db.close()

        run(_test())


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all test classes manually (for environments without pytest)."""
    import traceback

    classes = [
        TestPerTraderPnl,
        TestZombieDetection,
        TestUniformityMonitor,
        TestBestTraderScoring,
        TestFollowedTraders,
    ]

    passed = 0
    failed = 0

    for cls in classes:
        instance = cls()
        for method_name in dir(instance):
            if not method_name.startswith('test_'):
                continue
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL: {cls.__name__}.{method_name}")
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
