"""Pin: pool acquire timeout error includes operator-actionable
discriminators (pool_id, checked_out, idle) and a remediation hint.

The message previously gave only ``max_size`` and ``timeout`` — an
operator paged on a 3am ``DqliteConnectionError("Timed out waiting
for a connection from the pool")`` could not tell whether the cause
was a leaking application (checked_out at max, idle 0) or a slow
cluster (checked_out below max, idle 0). Both have the same parameters
in the message but very different remediations.

Pin the new fields against regression so a future refactor can't
accidentally drop them.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


def _make_full_pool_at_capacity() -> ConnectionPool:
    """Build a minimal ConnectionPool with no available connections
    so an acquire times out immediately."""
    import os

    pool = ConnectionPool.__new__(ConnectionPool)
    pool._closed = False
    pool._max_size = 5
    pool._size = 5  # checked-out + reserved == max_size
    pool._timeout = 0.01
    pool._pool = asyncio.Queue()  # no connections in the queue
    pool._lock = asyncio.Lock()
    pool._closed_event = None
    pool._addresses = ["host:9001"]
    pool._creator_pid = os.getpid()
    return pool


async def _acquire_with_timeout(pool: ConnectionPool) -> None:
    async with pool.acquire():
        pass


@pytest.mark.asyncio
async def test_acquire_timeout_message_includes_pool_id() -> None:
    pool = _make_full_pool_at_capacity()
    with pytest.raises(DqliteConnectionError) as exc_info:
        await _acquire_with_timeout(pool)
    text = str(exc_info.value)
    assert f"pool_id={id(pool)}" in text


@pytest.mark.asyncio
async def test_acquire_timeout_message_includes_checked_out_and_idle() -> None:
    pool = _make_full_pool_at_capacity()
    with pytest.raises(DqliteConnectionError) as exc_info:
        await _acquire_with_timeout(pool)
    text = str(exc_info.value)
    assert "checked_out=" in text
    assert "idle=" in text


@pytest.mark.asyncio
async def test_acquire_timeout_message_includes_remediation_hint() -> None:
    """The hint discriminates leak vs slow-cluster — pin so a future
    edit doesn't strip it."""
    pool = _make_full_pool_at_capacity()
    with pytest.raises(DqliteConnectionError) as exc_info:
        await _acquire_with_timeout(pool)
    text = str(exc_info.value)
    assert "leaking" in text.lower() or "leak" in text.lower()


@pytest.mark.asyncio
async def test_acquire_timeout_message_keeps_max_size_and_timeout() -> None:
    """Existing fields must still be present — they are referenced
    by operators / runbooks."""
    pool = _make_full_pool_at_capacity()
    with pytest.raises(DqliteConnectionError) as exc_info:
        await _acquire_with_timeout(pool)
    text = str(exc_info.value)
    assert "max_size=5" in text
    assert "timeout=" in text
