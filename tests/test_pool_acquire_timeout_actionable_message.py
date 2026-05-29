"""Acquire-timeout error carries operator discriminators (pool_id, checked_out,
idle) plus a leak-vs-slow-cluster remediation hint, not just max_size/timeout."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


def _make_full_pool_at_capacity() -> ConnectionPool:
    """Pool with no available connections so acquire times out immediately."""
    import os

    pool = ConnectionPool.__new__(ConnectionPool)
    pool._closed = False
    pool._max_size = 5
    pool._size = 5
    pool._timeout = 0.01
    pool._pool = asyncio.Queue()
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
    """The hint discriminates leak vs slow-cluster."""
    pool = _make_full_pool_at_capacity()
    with pytest.raises(DqliteConnectionError) as exc_info:
        await _acquire_with_timeout(pool)
    text = str(exc_info.value)
    assert "leaking" in text.lower() or "leak" in text.lower()


@pytest.mark.asyncio
async def test_acquire_timeout_message_keeps_max_size_and_timeout() -> None:
    """Existing fields stay present — operators and runbooks reference them."""
    pool = _make_full_pool_at_capacity()
    with pytest.raises(DqliteConnectionError) as exc_info:
        await _acquire_with_timeout(pool)
    text = str(exc_info.value)
    assert "max_size=5" in text
    assert "timeout=" in text
