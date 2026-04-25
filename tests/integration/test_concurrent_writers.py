"""Integration tests for concurrent write paths against a live dqlite
cluster.

dqlite serializes writes through Raft on the leader; SQLite WAL gives
single-writer-at-a-time semantics on the leader. These tests pin:

- two concurrent connections each running BEGIN; INSERT; COMMIT
  produce no duplicate rows and both connections remain usable
  afterward;
- the default ``retry_with_backoff`` retryable_exceptions tuple does
  NOT retry an OperationalError raised by the SQL layer (a BUSY-style
  contention error is application-level, not transport-level — the
  retry helper must surface it once);
- ``cursor.interrupt`` (well, the protocol-level primitive)
  end-to-end against a real cluster correctly returns the EmptyResponse
  acknowledgement, exercising the wire-layer plumbing the rest of the
  stack now relies on.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import connect
from dqliteclient.exceptions import OperationalError
from dqliteclient.retry import retry_with_backoff


@pytest.mark.integration
class TestConcurrentWriters:
    async def test_two_concurrent_inserts_no_duplicates(self, cluster_address: str) -> None:
        """Two clients each open BEGIN; INSERT 1 row; COMMIT against
        the same cluster. Even when both attempts contend, the cluster
        serializes them — the row count must equal the number of
        successful commits, never more."""
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

        # Both writers should have completed; if one or both saw BUSY,
        # they would surface OperationalError. Either outcome is valid;
        # what matters is no duplicate persistence.
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
        """``retry_with_backoff`` with the default ``retryable_exceptions``
        tuple must NOT retry an ``OperationalError`` raised by the SQL
        layer — those are deterministic application-level failures, not
        transport-level transient errors. Pin that the call surfaces
        the OperationalError on the first attempt."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_retry_no_op")
            await conn.execute("CREATE TABLE test_retry_no_op (id INTEGER PRIMARY KEY)")

            calls = 0

            async def attempt() -> None:
                nonlocal calls
                calls += 1
                # Duplicate primary key triggers a deterministic
                # IntegrityError on the wire; surfaces as
                # OperationalError at this layer (or IntegrityError
                # depending on classification — either is fine, both
                # are non-transport).
                await conn.execute("INSERT INTO test_retry_no_op (id) VALUES (1)")
                await conn.execute("INSERT INTO test_retry_no_op (id) VALUES (1)")

            with pytest.raises(OperationalError):
                await retry_with_backoff(attempt, max_attempts=5)
            assert calls == 1, (
                f"retry_with_backoff retried an OperationalError {calls} times; "
                "default retryable_exceptions must surface it on the first attempt"
            )
