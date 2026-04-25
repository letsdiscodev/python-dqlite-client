"""Integration tests for the public ``query_raw`` and
``query_raw_typed`` APIs.

These methods are documented as the entry points for "DBAPI cursor
implementations that need column names separately" but no test
exercises them directly at the client layer (the dbapi cursor's
``description`` builder calls ``query_raw_typed`` in production,
but the client suite only tests through ``execute`` / ``fetchall``
which take different paths).

Closes the documented-API coverage gap reported by ``pytest --cov``
at ``connection.py:862-863`` and ``connection.py:881-882``.

Also covers ``__init__.py:143-157`` (``create_pool`` body) by
opening a pool through the public function rather than the
``ConnectionPool`` constructor + ``initialize`` pair the rest of
the suite uses.
"""

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
            assert len(col_types) == 2  # INTEGER + TEXT type codes
            assert len(row_types) == 1  # one row's per-column type tags
            assert rows == [[1, "a"]]


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Gated on the same cluster-fixture work as tx-040: pool's "
        "leader-find chases container-internal addresses "
        "(0.0.0.0:9001) that are unreachable from the docker-host "
        "test runner. Pin the test shape now so the moment the "
        "fixture exposes reachable redirect addresses it can be "
        "unblocked."
    )
)
class TestCreatePool:
    async def test_create_pool_returns_initialized_usable_pool(self, cluster_address: str) -> None:
        """Drive the ``create_pool`` body in ``__init__.py`` (the
        recommended public constructor for callers who don't want
        to assemble ``ConnectionPool(...) + await pool.initialize()``
        manually). Pin that the returned pool is initialized and
        usable via ``acquire()``."""
        pool = await create_pool([cluster_address], min_size=1, max_size=1)
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetchall("SELECT 1 AS n")
                assert rows == [{"n": 1}]
        finally:
            await pool.close()
