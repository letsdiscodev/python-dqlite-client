"""Concurrent write paths against a live cluster: two concurrent BEGIN/INSERT/COMMIT
produce no duplicate rows, and default ``retry_with_backoff`` does NOT retry a SQL-layer
OperationalError (application-level, not transport-transient — surface it once)."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import connect
from dqliteclient.exceptions import OperationalError
from dqliteclient.retry import retry_with_backoff


@pytest.mark.integration
class TestConcurrentWriters:
    async def test_two_concurrent_inserts_no_duplicates(self, cluster_address: str) -> None:
        """Two clients each BEGIN/INSERT/COMMIT; persisted rows must equal successful commits."""
        async with await connect(cluster_address) as setup:
            await setup.execute("DROP TABLE IF EXISTS test_concurrent_w")
            await setup.execute(
                "CREATE TABLE test_concurrent_w (id INTEGER PRIMARY KEY, marker TEXT)"
            )

        results: list[tuple[str, BaseException | None]] = []

        async def writer(marker: str) -> None:
            try:
                async with await connect(cluster_address) as conn, conn.transaction():
                    await conn.execute(
                        "INSERT INTO test_concurrent_w (marker) VALUES (?)",
                        [marker],
                    )
                results.append((marker, None))
            except BaseException as e:
                results.append((marker, e))

        await asyncio.gather(writer("A"), writer("B"))

        # Either writer may surface OperationalError on BUSY; what matters is no dup persistence.
        async with await connect(cluster_address) as conn:
            rows = await conn.fetch("SELECT marker FROM test_concurrent_w ORDER BY marker")
            successful_markers = sorted(m for m, e in results if e is None)
            persisted_markers = sorted(r["marker"] for r in rows)
            assert successful_markers == persisted_markers, (
                f"Persisted markers {persisted_markers} do not match successful "
                f"writes {successful_markers}; results={results}"
            )

    async def test_retry_with_backoff_default_does_not_retry_operational_error(
        self, cluster_address: str
    ) -> None:
        """Default ``retry_with_backoff`` must NOT retry a SQL-layer OperationalError;
        it surfaces on the first attempt (deterministic, not transport-transient)."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_retry_no_op")
            await conn.execute("CREATE TABLE test_retry_no_op (id INTEGER PRIMARY KEY)")

            calls = 0

            async def attempt() -> None:
                nonlocal calls
                calls += 1
                # Duplicate primary key: deterministic non-transport error.
                await conn.execute("INSERT INTO test_retry_no_op (id) VALUES (1)")
                await conn.execute("INSERT INTO test_retry_no_op (id) VALUES (1)")

            with pytest.raises(OperationalError):
                await retry_with_backoff(attempt, max_attempts=5)
            assert calls == 1, (
                f"retry_with_backoff retried an OperationalError {calls} times; "
                "default retryable_exceptions must surface it on the first attempt"
            )
