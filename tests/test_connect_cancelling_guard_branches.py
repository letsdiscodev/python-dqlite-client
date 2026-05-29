"""Branch coverage for the ``Task.cancelling()`` guard in connect()'s pending-drain arm.

cancelling()==0 means our own ``pending.cancel()`` (swallow and proceed);
cancelling()>0 means an outer cancel is pending (re-raise).
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

import dqliteclient.connection as conn_mod
from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_connect_swallows_inner_pending_cancel_alone() -> None:
    """With no outer cancel, connect()'s own ``pending.cancel()`` is swallowed and proceeds."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)

    started = asyncio.Event()

    async def slow_drain() -> None:
        started.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(60)

    prior = asyncio.get_running_loop().create_task(slow_drain())
    await started.wait()
    conn._pending_drain = prior

    # Stub open_connection to fail promptly; we only need to prove it was reached.
    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int, **_kwargs: object):
        open_connection_called.append((host, port))
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]

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

    asyncio.open_connection = fake_open  # type: ignore[assignment]
    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    original_abort = DqliteConnection._abort_protocol
    DqliteConnection._abort_protocol = fake_abort

    try:
        with pytest.raises(RuntimeError, match="synthetic stop"):
            await conn.connect()
    finally:
        asyncio.open_connection = real_open
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = original_abort
        if not prior.done():
            prior.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await prior

    assert open_connection_called, (
        "connect() must consume its own ``pending.cancel()`` and "
        "proceed to open_connection — Task.cancelling() must report "
        "0 when no outer cancel is pending."
    )


@pytest.mark.asyncio
async def test_connect_with_no_pending_drain_skips_drain_block_entirely() -> None:
    """When ``_pending_drain`` is None the drain block is skipped and connect() proceeds."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)
    assert conn._pending_drain is None

    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int, **_kwargs: object):
        open_connection_called.append((host, port))
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]

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

    asyncio.open_connection = fake_open  # type: ignore[assignment]
    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    original_abort = DqliteConnection._abort_protocol
    DqliteConnection._abort_protocol = fake_abort

    try:
        with pytest.raises(RuntimeError, match="synthetic stop"):
            await conn.connect()
    finally:
        asyncio.open_connection = real_open
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = original_abort

    assert open_connection_called, (
        "connect() with no prior pending drain must reach open_connection"
    )


@pytest.mark.asyncio
async def test_connect_swallows_already_done_pending_drain() -> None:
    """An already-done prior drain skips the cancel-and-await dance; connect() proceeds."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)

    async def already_done() -> None:
        return None

    prior = asyncio.get_running_loop().create_task(already_done())
    await prior
    assert prior.done()
    conn._pending_drain = prior

    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int, **_kwargs: object):
        open_connection_called.append((host, port))
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]

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

    asyncio.open_connection = fake_open  # type: ignore[assignment]
    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    original_abort = DqliteConnection._abort_protocol
    DqliteConnection._abort_protocol = fake_abort

    try:
        with pytest.raises(RuntimeError, match="synthetic stop"):
            await conn.connect()
    finally:
        asyncio.open_connection = real_open
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = original_abort

    assert open_connection_called, (
        "connect() with an already-done prior drain must skip the "
        "cancel-and-await dance and reach open_connection"
    )
    assert conn._pending_drain is None, (
        "connect() must clear ``_pending_drain`` after retiring it, "
        "even if it was already done (so a future ``_invalidate`` "
        "doesn't see a stale reference)."
    )
