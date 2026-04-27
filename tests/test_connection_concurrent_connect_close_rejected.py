"""Pin: concurrent ``connect()`` / ``close()`` calls on a single
``DqliteConnection`` are rejected via the synchronous ``_in_use``
claim instead of racing through the awaits and orphaning a transport.

The pool serializes via ``_lock``; direct-DqliteConnection users
(calling ``await conn.connect()`` directly) hit the in-use guard
when a sibling task / coroutine attempts the same operation
concurrently. Without the synchronous claim, both calls passed
``_check_in_use()`` (both saw ``_in_use=False``), then both
proceeded, building duplicate transports.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_concurrent_connect_calls_reject_one_via_in_use_guard() -> None:
    """A second ``connect()`` running concurrently with a first must
    raise ``InterfaceError`` from ``_check_in_use``, not race through."""
    conn = DqliteConnection("localhost:9001", timeout=1.0)

    # Stub open_connection so connect() is deterministic. Make it slow
    # so the second concurrent call lands while the first is in flight.
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_open(host: str, port: int):
        started.set()
        await proceed.wait()
        # Construct a (reader, writer) that fail handshake quickly.
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    class _StubProto:
        def __init__(self, *a: object, **kw: object) -> None:
            self._client_id = 0
            self._writer = MagicMock()

        async def handshake(self) -> None:
            raise RuntimeError("synthetic stop")

        async def wait_closed(self) -> None:
            return None

        def close(self) -> None:
            return None

    async def fake_abort(self: object) -> None:
        return None

    import dqliteclient.connection as conn_mod

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_open = asyncio.open_connection
    real_abort = DqliteConnection._abort_protocol

    asyncio.open_connection = slow_open  # type: ignore[assignment]
    conn_mod.DqliteProtocol = _StubProto  # type: ignore[assignment,attr-defined]
    DqliteConnection._abort_protocol = fake_abort
    try:
        first = asyncio.get_running_loop().create_task(conn.connect())
        await started.wait()
        # Now first is suspended inside open_connection. Attempt a second.
        with pytest.raises(InterfaceError, match="another operation"):
            await conn.connect()
        # Let first complete (it raises the synthetic RuntimeError).
        proceed.set()
        with pytest.raises(RuntimeError, match="synthetic stop"):
            await first
    finally:
        asyncio.open_connection = real_open
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = real_abort


@pytest.mark.asyncio
async def test_concurrent_close_and_connect_rejected_via_in_use_guard() -> None:
    """``close()`` running concurrently with ``connect()`` on the same
    instance must produce an ``InterfaceError`` from one of the two
    calls. The synchronous ``_in_use=True`` claim in both methods
    ensures the second-arriving caller hits ``_check_in_use``."""
    conn = DqliteConnection("localhost:9001", timeout=1.0)

    # Pre-populate a fake _protocol so close() has work to do.
    fake_proto = MagicMock()
    fake_proto.close = MagicMock()
    fake_proto.wait_closed = AsyncMock()
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_wait_closed() -> None:
        started.set()
        await proceed.wait()

    fake_proto.wait_closed = slow_wait_closed
    conn._protocol = fake_proto

    close_task = asyncio.get_running_loop().create_task(conn.close())
    await started.wait()
    # Now close() is suspended on wait_closed. A concurrent connect()
    # should be rejected.
    with pytest.raises(InterfaceError, match="another operation"):
        await conn.connect()
    proceed.set()
    await close_task
