"""Pin: ``DqliteConnection._abort_protocol`` shields its inner drain
across outer cancels so the underlying StreamReader task is not
orphaned.

An outer cancel landing while ``wait_closed`` is in flight
previously cancelled the drain without awaiting the inner task to
completion — surfacing as ``"Task was destroyed but it is pending"``
warnings at GC. The shield+done-callback discipline (mirroring the
sibling ``_connect_impl`` finally arm and
``ClusterClient.open_admin_connection``'s drain) keeps the inner
task alive and observed.
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


def _make_connection_with_stub_protocol(
    wait_closed_blocker: asyncio.Event,
) -> tuple[DqliteConnection, MagicMock]:
    """Build a connection whose ``_protocol.wait_closed`` blocks on
    the supplied event so the test owns the drain timing. The
    ``close()`` synchronous method is a no-op for the stub.
    """
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._close_timeout = 5.0
    conn._address = "localhost:9001"
    proto = MagicMock()
    proto.close = MagicMock()

    async def _slow_wait_closed() -> None:
        await wait_closed_blocker.wait()

    proto.wait_closed = AsyncMock(side_effect=_slow_wait_closed)
    conn._protocol = proto
    return conn, proto


async def test_abort_protocol_outer_cancel_does_not_orphan_inner_drain() -> None:
    """Schedule ``_abort_protocol`` inside an ``asyncio.timeout`` that
    fires before the inner drain completes. The shield must keep the
    inner task alive to completion (via the done-callback discipline)
    so no ``ResourceWarning`` / ``Task was destroyed`` warning lands.
    """
    blocker = asyncio.Event()
    conn, proto = _make_connection_with_stub_protocol(blocker)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await conn._abort_protocol()
        # Let the inner drain finish so the done-callback observes it.
        blocker.set()
        # Yield enough loop ticks for the scheduled inner_drain to
        # run its body and fire the done-callback.
        for _ in range(10):
            await asyncio.sleep(0)
        # Force GC so any pending-task warnings would surface.
        gc.collect()

    pending_task_warnings = [
        str(w.message)
        for w in caught
        if "pending" in str(w.message).lower() or "destroyed" in str(w.message).lower()
    ]
    assert pending_task_warnings == [], (
        f"outer cancel must not orphan the inner drain task; got {pending_task_warnings!r}"
    )
    # The sync ``proto.close()`` synchronous tear-down still ran
    # regardless of the outer cancel.
    proto.close.assert_called_once_with()


async def test_abort_protocol_with_no_protocol_returns_immediately() -> None:
    """If ``_protocol`` is None (already aborted / never connected),
    the method is a no-op. Preserved across the shield migration.
    """
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._close_timeout = 5.0
    conn._address = "localhost:9001"
    conn._protocol = None
    await conn._abort_protocol()
    # ``conn._protocol`` was None to begin with; the early return
    # leaves it untouched.
    assert conn._protocol is None
