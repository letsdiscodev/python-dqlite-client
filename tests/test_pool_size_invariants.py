"""Concurrent-access invariants of ConnectionPool.

Verifies that _size never exceeds max_size under concurrent acquires,
even when _create_connection is slow (simulating TCP handshake
latency) — a regression would be the reservation pattern losing
ordering between check and increment."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from dqliteclient.pool import ConnectionPool


async def _make_conn() -> MagicMock:
    mock = MagicMock()
    mock.is_connected = True
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock._in_transaction = False
    mock._pool_released = False
    mock._tx_owner = None
    mock.execute = AsyncMock()
    return mock


class TestPoolSizeInvariants:
    async def test_concurrent_acquires_respect_max_size(self) -> None:
        """10 concurrent acquires with max_size=3 must create at most 3.

        With the pre-fix implementation that held the lock across the
        network call, this would pass — but slowly. The new
        reservation pattern drops the lock for the create; this test
        ensures it still respects the cap.
        """
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=3, timeout=5.0)

        create_count = 0
        peak_size = 0

        async def slow_create(**kwargs):
            nonlocal create_count, peak_size
            create_count += 1
            peak_size = max(peak_size, pool._size)
            # Simulate TCP handshake latency; allow other tasks to run.
            for _ in range(5):
                await asyncio.sleep(0)
            return await _make_conn()

        with patch.object(pool._cluster, "connect", side_effect=slow_create):

            async def borrow() -> None:
                async with pool.acquire() as c:
                    assert c is not None

            await asyncio.gather(*(borrow() for _ in range(10)))

        assert create_count == 3, f"expected 3 creates (max_size), got {create_count}"
        assert peak_size <= pool._max_size, (
            f"_size exceeded max_size: peak={peak_size}, max={pool._max_size}"
        )
        await pool.close()

    async def test_create_failure_rolls_back_reservation(self) -> None:
        """If _create_connection raises, the reservation must be released
        so a subsequent acquire can succeed."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1, timeout=1.0)

        call_count = 0

        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("boom")
            return await _make_conn()

        with patch.object(pool._cluster, "connect", side_effect=flaky_create):

            async def borrow() -> None:
                async with pool.acquire() as c:
                    assert c is not None

            # First call fails — reservation must be released.
            import pytest

            with pytest.raises(OSError):
                await borrow()
            assert pool._size == 0, f"reservation leaked after create failure; _size={pool._size}"

            # Second call should succeed.
            await borrow()

        await pool.close()

    async def test_drained_idle_releases_size(self) -> None:
        """_drain_idle must decrement _size by exactly the number of drained connections."""
        pool = ConnectionPool(["localhost:9001"], min_size=3, max_size=5, timeout=1.0)

        async def ok_create(**kwargs):
            return await _make_conn()

        with patch.object(pool._cluster, "connect", side_effect=ok_create):
            await pool.initialize()
            assert pool._size == 3

            await pool._drain_idle()
            assert pool._size == 0, f"drain_idle left _size={pool._size}"

        await pool.close()
