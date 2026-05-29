"""Concurrent-access invariants: _size never exceeds max_size under
concurrent acquires even with a slow _create_connection."""

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
        """10 concurrent acquires with max_size=3 must create at most 3,
        even though the reservation pattern drops the lock for the create."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=3, timeout=5.0)

        create_count = 0
        peak_size = 0

        async def slow_create(**kwargs):
            nonlocal create_count, peak_size
            create_count += 1
            peak_size = max(peak_size, pool._size)
            # Simulate handshake latency; let other tasks run.
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
        """A failed _create_connection releases the reservation so a later acquire succeeds."""
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

            import pytest

            with pytest.raises(OSError):
                await borrow()
            assert pool._size == 0, f"reservation leaked after create failure; _size={pool._size}"

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
