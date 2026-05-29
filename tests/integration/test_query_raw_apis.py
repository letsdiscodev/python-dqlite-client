"""Integration tests for the public ``query_raw`` / ``query_raw_typed`` and
``create_pool`` APIs."""

from __future__ import annotations

import pytest

from dqliteclient import connect, create_pool


@pytest.mark.integration
class TestQueryRawApis:
    async def test_query_raw_returns_columns_and_rows(self, cluster_address: str) -> None:
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS query_raw_t")
            await conn.execute("CREATE TABLE query_raw_t (n INTEGER, s TEXT)")
            await conn.execute("INSERT INTO query_raw_t (n, s) VALUES (?, ?)", [1, "a"])
            await conn.execute("INSERT INTO query_raw_t (n, s) VALUES (?, ?)", [2, "b"])

            cols, rows = await conn.query_raw("SELECT n, s FROM query_raw_t ORDER BY n")
            assert cols == ["n", "s"]
            assert rows == [[1, "a"], [2, "b"]]

    async def test_query_raw_typed_returns_columns_types_row_types_rows(
        self, cluster_address: str
    ) -> None:
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS query_raw_typed_t")
            await conn.execute("CREATE TABLE query_raw_typed_t (n INTEGER, s TEXT)")
            await conn.execute("INSERT INTO query_raw_typed_t (n, s) VALUES (?, ?)", [1, "a"])

            cols, col_types, row_types, rows = await conn.query_raw_typed(
                "SELECT n, s FROM query_raw_typed_t"
            )
            assert cols == ["n", "s"]
            assert len(col_types) == 2
            assert len(row_types) == 1
            assert rows == [[1, "a"]]


@pytest.mark.integration
class TestCreatePool:
    async def test_create_pool_returns_initialized_usable_pool(self, cluster_address: str) -> None:
        """``create_pool`` returns an initialized pool usable via ``acquire()``."""
        pool = await create_pool([cluster_address], min_size=1, max_size=1)
        try:
            async with pool.acquire() as conn:
                # fetchall returns positional row lists, not dicts.
                rows = await conn.fetchall("SELECT 1 AS n")
                assert rows == [[1]]
        finally:
            await pool.close()
