"""Pin: ``acquire`` re-checks a freshly dialed conn's ``is_connected`` before
yielding it, so a peer that sent FIN/RST between handshake and return is
rejected with ``DqliteConnectionError`` instead of failing on first query.
The gate is is_connected-only (no _socket_looks_dead) to avoid false positives
on MagicMock(spec=DqliteConnection) which lacks a mocked _protocol."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_rejects_freshly_dialed_dead_conn() -> None:
    """A freshly dialed conn reporting is_connected=False must be rejected with
    DqliteConnectionError, not yielded to the caller."""
    pool = ConnectionPool.__new__(ConnectionPool)
    # Slots the acquire path reads; pool is in "needs a fresh slot" state.
    pool._closed = False
    pool._closed_flag = [False]
    pool._max_size = 5
    pool._size = 0
    pool._reserved_flag = [False]
    pool._timeout = 5.0
    pool._lock = asyncio.Lock()
    pool._pool = asyncio.Queue(maxsize=5)
    pool._closed_event = asyncio.Event()
    pool._close_done = None
    pool._drain_complete = False
    pool._finalizer = None
    pool._creator_pid = os.getpid()

    fresh = MagicMock()
    fresh.is_connected = False
    fresh.close = AsyncMock()
    fresh._pool_released = False
    fresh.address = "test:9001"

    async def _fake_create(*args: Any, **kwargs: Any) -> Any:
        return fresh

    pool._create_connection = _fake_create

    try:
        with pytest.raises(DqliteConnectionError, match="is_connected=False"):
            async with pool.acquire():
                pytest.fail(
                    "acquire() yielded a freshly-dialed dead connection; "
                    "the recheck at checkout time did not fire"
                )
        fresh.close.assert_called()  # dead conn closed, reservation released
    finally:
        # Mark closed so the finalizer's GC warning does not fire on teardown.
        pool._closed = True
        pool._closed_flag[0] = True
