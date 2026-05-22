"""Pin: ``DqliteProtocol._send`` and ``DqliteProtocol._read_data``
use ``asyncio.timeout`` cancel-scope semantics rather than
``asyncio.wait_for``.

``asyncio.wait_for(fut, timeout)`` cancels the inner future on outer
cancel AND discards its result. ``async with asyncio.timeout(...):
await fut`` uses cancel-scope semantics: the inner future's bytes
(``_read_data``) or completion signal (``_send``) is observable to
the caller's ``try / except CancelledError`` handler.

The migration matches the sibling dial / connect / admin discipline
already established at four sites in ``cluster.py`` and
``connection.py``. ``_read_data`` is the load-bearing site (returns
bytes); ``_send`` is sibling-parity defence (returns ``None`` so the
result-discard concern does not apply today).

The diagnostic shape is preserved: an inner timeout still surfaces
as ``DqliteConnectionError(...)`` via the ``TimeoutError`` arm.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.protocol import DqliteProtocol


def _make_protocol(reader: MagicMock, writer: MagicMock) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._reader = reader
    proto._writer = writer
    proto._timeout = 0.5
    proto._read_timeout = 0.5
    proto._address = "localhost:9001"
    proto._heartbeat_timeout = 0
    return proto


@pytest.mark.asyncio
async def test_send_timeout_surfaces_as_dqlite_connection_error() -> None:
    """Diagnostic-preservation pin: a slow ``drain()`` produces the
    documented ``DqliteConnectionError`` shape, identical to the
    pre-migration ``asyncio.wait_for`` arm.
    """
    writer = MagicMock()

    async def _slow_drain() -> None:
        await asyncio.sleep(10)

    writer.drain = AsyncMock(side_effect=_slow_drain)
    proto = _make_protocol(MagicMock(), writer)
    proto._timeout = 0.01

    with pytest.raises(DqliteConnectionError, match=r"Write timeout"):
        await proto._send()


@pytest.mark.asyncio
async def test_read_data_timeout_surfaces_as_dqlite_connection_error() -> None:
    """Diagnostic-preservation pin for ``_read_data``."""
    reader = MagicMock()

    async def _slow_read(_n: int) -> bytes:
        await asyncio.sleep(10)
        return b""

    reader.read = _slow_read
    proto = _make_protocol(reader, MagicMock())
    proto._read_timeout = 0.01

    with pytest.raises(DqliteConnectionError, match=r"timed out"):
        await proto._read_data()


@pytest.mark.asyncio
async def test_send_outer_cancel_propagates_as_cancel_not_dqlite_error() -> None:
    """Cancel-scope semantics: an outer ``task.cancel()`` lands on
    the awaiter as ``CancelledError``, not the timeout-shaped
    ``DqliteConnectionError`` that ``asyncio.wait_for`` would map.
    """
    writer = MagicMock()
    drain_started = asyncio.Event()
    cancel_now = asyncio.Event()

    async def _drain_then_block() -> None:
        drain_started.set()
        try:
            await cancel_now.wait()
        except asyncio.CancelledError:
            raise

    writer.drain = AsyncMock(side_effect=_drain_then_block)
    proto = _make_protocol(MagicMock(), writer)
    proto._timeout = 60.0  # long timeout so only the outer cancel fires

    async def run() -> None:
        await proto._send()

    task = asyncio.create_task(run())
    await drain_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_read_data_outer_cancel_propagates_as_cancel() -> None:
    """Cancel-scope semantics on ``_read_data``: an outer cancel
    surfaces as ``CancelledError``, not ``DqliteConnectionError``.
    """
    reader = MagicMock()
    read_started = asyncio.Event()
    cancel_now = asyncio.Event()

    async def _read_then_block(_n: int) -> bytes:
        read_started.set()
        await cancel_now.wait()
        return b""

    reader.read = _read_then_block
    proto = _make_protocol(reader, MagicMock())
    proto._read_timeout = 60.0

    async def run() -> None:
        await proto._read_data()

    task = asyncio.create_task(run())
    await read_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
