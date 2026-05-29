"""Pin: ``DqliteProtocol._send`` / ``_read_data`` use
``asyncio.timeout`` cancel-scope semantics, not ``asyncio.wait_for``
(which discards the inner result on outer-cancel). An inner timeout
still surfaces as ``DqliteConnectionError`` via the ``TimeoutError`` arm.
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
    """A slow ``drain()`` surfaces as ``DqliteConnectionError``."""
    writer = MagicMock()

    async def _slow_drain() -> None:
        await asyncio.sleep(10)

    writer.drain = AsyncMock(side_effect=_slow_drain)
    proto = _make_protocol(MagicMock(), writer)
    proto._timeout = 0.01

    with pytest.raises(DqliteConnectionError, match=r"Write timeout"):
        await proto._send(b"")


@pytest.mark.asyncio
async def test_read_data_timeout_surfaces_as_dqlite_connection_error() -> None:
    """A slow ``read()`` surfaces as ``DqliteConnectionError``."""
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
    """An outer cancel surfaces as ``CancelledError``, not
    ``DqliteConnectionError``."""
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
        await proto._send(b"")

    task = asyncio.create_task(run())
    await drain_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_read_data_outer_cancel_propagates_as_cancel() -> None:
    """An outer cancel on ``_read_data`` surfaces as ``CancelledError``."""
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
