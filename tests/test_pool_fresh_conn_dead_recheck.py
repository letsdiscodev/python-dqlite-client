"""Pin: ``ConnectionPool.acquire`` re-checks the freshly dialed
connection's ``is_connected`` flag before yielding it to the caller.

Symmetric (but narrower) with the queue-dequeue ``is_connected /
_socket_looks_dead`` check at the top of the acquire path. Without
the re-check, a peer that sent FIN/RST between handshake completion
and ``_create_connection``'s return (rare but possible under
firewall idle-timeout / peer Raft flip / peer process crash) would
surface as an opaque transport error on the caller's first query.
The pin asserts the rejection lands at checkout time with a clean
``DqliteConnectionError``. The narrower ``is_connected``-only gate
(no ``_socket_looks_dead`` peek) deliberately avoids false-positives
on the standard mock pattern used throughout the pool test suite
where ``MagicMock(spec=DqliteConnection)`` does not expose a mocked
``_protocol`` attribute.
"""

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
    """Patch ``_create_connection`` to return a conn whose
    ``is_connected`` reports False (mimics a peer that sent FIN
    between handshake and return). Acquire must reject it with
    DqliteConnectionError, NOT yield it to the caller's context
    manager body."""
    pool = ConnectionPool.__new__(ConnectionPool)
    # Skip __init__ — set the slots the acquire path reads. The pool
    # is in "needs a fresh slot" state (empty queue, room to grow).

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

    # Fresh dial returns a "dead" connection — is_connected=False is
    # the definitive signal the pool's checkout re-check looks for.
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
        # The dead conn was closed; the reservation was released.
        fresh.close.assert_called()
    finally:
        # Mark the pool closed so the finalizer-emitted warning at GC
        # does not fire and so any leftover loop-bound primitives are
        # detached cleanly when the test's event loop tears down.
        pool._closed = True
        pool._closed_flag[0] = True
