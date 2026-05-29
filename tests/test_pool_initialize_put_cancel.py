"""Pool.initialize must keep _size accounting consistent on a publish abort:
release only uncommitted slots and close unqueued survivors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_publish_failure_releases_reservations_and_closes_survivors() -> None:
    """Phase C publish failure (synthetic ``put_nowait`` error): every reserved
    slot is released, every survivor closed, and the failure propagates."""
    pool = ConnectionPool(["localhost:19001"], min_size=5, max_size=5, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(5):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)

    async def _create() -> object:
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    # Raise on the third put_nowait so Phase C routes ALL successes through
    # the close helper (publish-failure cleanup is all-or-nothing).
    original_put_nowait = pool._pool.put_nowait
    put_call_count = 0

    def _put_nowait(conn: object) -> None:
        nonlocal put_call_count
        put_call_count += 1
        if put_call_count == 3:
            raise RuntimeError("synthetic Phase C failure")
        original_put_nowait(conn)  # type: ignore[arg-type]

    pool._pool.put_nowait = _put_nowait  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="synthetic Phase C failure"):
        await pool.initialize()

    # On Phase C failure all successes are closed and the queue ends empty.
    assert pool._size == 0, f"_size should be 0, got {pool._size}"
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"survivor {i} was not closed"
    assert pool._initialized is False


@pytest.mark.asyncio
async def test_initialize_clean_success_leaves_size_at_min_size() -> None:
    """Happy path: the ``unqueued`` bookkeeping must not over-release."""
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)

    async def _create() -> object:
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    await pool.initialize()

    assert pool._pool.qsize() == 3
    assert pool._size == 3
    for m in mocks:
        m.close.assert_not_awaited()
