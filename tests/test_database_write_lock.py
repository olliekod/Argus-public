from __future__ import annotations

import asyncio

import pytest

from src.core.database import Database


@pytest.mark.asyncio
async def test_database_initializes_write_lock(tmp_path) -> None:
    db = Database(str(tmp_path / "argus.db"))
    await db.connect()
    try:
        assert db._write_lock is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_database_serializes_execute_and_execute_many(tmp_path) -> None:
    db = Database(str(tmp_path / "argus.db"))
    await db.connect()
    try:
        await db.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)")

        async def write_one(i: int) -> None:
            await db.execute("INSERT INTO t(v) VALUES (?)", (i,))

        async def write_many(start: int) -> None:
            await db.execute_many(
                "INSERT INTO t(v) VALUES (?)",
                [(start,), (start + 1,), (start + 2,)],
            )

        await asyncio.gather(write_one(1), write_many(10), write_one(2), write_many(20))
        rows = await db.fetch_all("SELECT COUNT(*) AS c FROM t")
        assert rows[0][0] == 8
    finally:
        await db.close()
