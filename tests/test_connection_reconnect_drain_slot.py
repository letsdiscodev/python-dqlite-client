"""connect() must cancel-and-await a prior ``_pending_drain`` so a later
``_invalidate`` cannot clobber an un-awaited drain task (single-ref discipline)."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_connect_cancels_prior_pending_drain() -> None:
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=1.0)

    # A still-running drain task from a prior invalidate.
    ran: list[str] = []

    async def never_completing() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            ran.append("cancelled")
            raise

    prior = asyncio.get_running_loop().create_task(never_completing())
    # Let it start so cancel has something to deliver to.
    await asyncio.sleep(0)
    conn._pending_drain = prior

    called: list[object] = []

    async def fake_open_connection(host: str, port: int, **_kwargs: object):
        called.append((host, port))
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    import dqliteclient.connection as conn_mod

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_open = asyncio.open_connection

    class _StubProtocol:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._client_id = 0
            self._writer = args[1] if len(args) >= 2 else MagicMock()

        async def handshake(self) -> None:
            raise RuntimeError("synthetic stop")

        async def wait_closed(self) -> None:
            return None

        def close(self) -> None:
            return None

    async def fake_abort(self: object) -> None:
        return None

    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    asyncio.open_connection = fake_open_connection  # type: ignore[assignment]
    original_abort = DqliteConnection._abort_protocol
    DqliteConnection._abort_protocol = fake_abort

    try:
        with pytest.raises(RuntimeError, match="synthetic stop"):
            await conn.connect()
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        asyncio.open_connection = real_open
        DqliteConnection._abort_protocol = original_abort

    assert prior.done(), (
        "connect() must cancel-and-await any prior pending drain so a "
        "subsequent _invalidate does not overwrite a live task reference"
    )
    with pytest.raises(asyncio.CancelledError):
        await prior


@pytest.mark.asyncio
async def test_connect_propagates_outer_cancel_during_pending_drain_retire() -> None:
    """An outer cancel delivered while connect() awaits the prior drain must
    propagate, not be swallowed (which would let connect() proceed to dial)."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)

    started = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_drain() -> None:
        started.set()
        try:
            await proceed.wait()
        except asyncio.CancelledError:
            # Yield so the outer cancel lands while connect() is awaiting us.
            await asyncio.sleep(0)
            raise

    prior = asyncio.get_running_loop().create_task(slow_drain())
    await started.wait()
    conn._pending_drain = prior

    # If connect() proceeds past the pending-retire, this mock fires —
    # that's the regression we guard against.
    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int, **_kwargs: object):
        open_connection_called.append((host, port))
        return MagicMock(), MagicMock()

    asyncio.open_connection = fake_open  # type: ignore[assignment]
    try:

        async def driver() -> None:
            await conn.connect()

        driver_task = asyncio.get_running_loop().create_task(driver())
        # Let driver enter connect() and reach ``await pending``.
        for _ in range(3):
            await asyncio.sleep(0)

        driver_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await driver_task
    finally:
        asyncio.open_connection = real_open
        if not prior.done():
            prior.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await prior

    assert not open_connection_called, (
        "connect() must not invoke open_connection after an outer "
        "task.cancel() delivered during the pending-drain await"
    )
