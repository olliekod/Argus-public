import sqlite3

from scripts.tastytrade_health_audit import _ensure_snapshot_table, _prune_snapshots_sql


def test_option_quote_snapshots_schema_creation_and_indexes(tmp_path):
    db_path = tmp_path / "snapshots.sqlite"
    _ensure_snapshot_table(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cols = conn.execute("PRAGMA table_info(option_quote_snapshots)").fetchall()
        col_names = [c[1] for c in cols]
        indexes = conn.execute("PRAGMA index_list(option_quote_snapshots)").fetchall()
        index_names = {row[1] for row in indexes}
    finally:
        conn.close()

    expected = [
        "id",
        "ts_utc",
        "provider",
        "underlying",
        "option_symbol",
        "expiry",
        "strike",
        "right",
        "bid",
        "ask",
        "mid",
        "event_ts",
        "recv_ts",
    ]
    assert col_names == expected
    assert "idx_option_quote_snapshots_underlying_ts" in index_names
    assert "idx_option_quote_snapshots_symbol_ts" in index_names
    assert "idx_option_quote_snapshots_provider" in index_names
    assert "idx_option_quote_snapshots_contract" in index_names


def test_prune_sql_respects_days():
    sql, params = _prune_snapshots_sql(14)
    assert "DELETE FROM option_quote_snapshots" in sql
    assert params == ("-14 days",)
