# Created by Oliver Meihls

import pytest

from src.core.database import Database


@pytest.mark.asyncio
async def test_get_kalshi_daily_pnl_history_filters_by_bot(tmp_path):
    db = Database(str(tmp_path / "kalshi.db"))
    await db.connect()
    try:
        await db.insert_kalshi_outcome(
            market_ticker="KX-1",
            strike=1.0,
            direction="yes",
            entry_price=0.4,
            exit_price=1.0,
            quantity=100,
            pnl=0.6,
            outcome="WON",
            final_avg=1.0,
            settled_at_ms=1700000000000,
            is_paper=True,
            bot_id="bot_a",
        )
        await db.insert_kalshi_outcome(
            market_ticker="KX-2",
            strike=1.0,
            direction="yes",
            entry_price=0.4,
            exit_price=0.0,
            quantity=100,
            pnl=-0.4,
            outcome="LOST",
            final_avg=0.0,
            settled_at_ms=1700000000000,
            is_paper=True,
            bot_id="bot_b",
        )

        hist_a = await db.get_kalshi_daily_pnl_history(days=10, bot_id="bot_a")
        hist_b = await db.get_kalshi_daily_pnl_history(days=10, bot_id="bot_b")

        assert len(hist_a) == 1
        assert hist_a[0]["pnl"] == pytest.approx(0.6)
        assert len(hist_b) == 1
        assert hist_b[0]["pnl"] == pytest.approx(-0.4)
    finally:
        await db.close()
