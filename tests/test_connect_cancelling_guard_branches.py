"""Branch coverage for ``DqliteConnection.connect()``'s
``Task.cancelling()`` guard inside the pending-drain ``except
asyncio.CancelledError`` arm.

The arm distinguishes two concrete cases:

1. ``Task.cancelling() == 0`` — the CancelledError came from our own
   ``pending.cancel()``; consume it and proceed with the connect.
2. ``Task.cancelling() > 0`` — an outer ``task.cancel()`` is pending
   on the current task; re-raise so the next checkpoint observes it.

The existing ``test_connection_reconnect_drain_slot.py`` covers
case 2 (outer cancel only). This file pins case 1 (inner cancel
only — connect must proceed) so a future refactor that drops the
``cancelling()`` check entirely (e.g., bare ``pass`` to "always
swallow", or bare ``raise`` to "always propagate", or
``cancelling() == 1`` instead of ``> 0``) is caught by at least one
test in the suite. Also adds a "no pending drain" baseline so the
common case (no prior invalidate) stays covered.
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
    """When ``connect()`` cancels the prior pending-drain via
    ``pending.cancel()`` and there is NO outer cancel against the
    current task, ``Task.cancelling()`` must report 0 — the resulting
    CancelledError is consumed (not re-raised) so the open_connection
    path actually executes.

    Inverts the existing outer-cancel pin: same fixtures, but no
    outer ``task.cancel()`` is delivered. ``open_connection`` MUST
    fire (proving connect() proceeded past the pending-drain block).
    """
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)

    # Long-running drain task — ``connect()`` will issue
    # ``pending.cancel()`` then ``await pending`` to drain it.
    started = asyncio.Event()

    async def slow_drain() -> None:
        started.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(60)

    prior = asyncio.get_running_loop().create_task(slow_drain())
    await started.wait()
    conn._pending_drain = prior

    # Stub ``open_connection`` to fail promptly so we don't need a
    # real server — we only need to prove it was reached.
    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int):
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
        # No outer cancel — the only CancelledError that can arise
        # comes from connect()'s own ``pending.cancel()``. The
        # ``cancelling() > 0`` guard must report 0 and let connect()
        # proceed to the open_connection call below.
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
    """Baseline pin: when ``_pending_drain`` is None (the common
    case — no prior invalidate landed) the cancel-guard arm is
    skipped entirely. open_connection must still execute. Catches a
    refactor that accidentally guards the drain block on the WRONG
    condition (e.g., ``if pending is None`` inverted)."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)
    assert conn._pending_drain is None

    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int):
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
    """If the prior pending-drain task is already ``done()``,
    ``connect()`` skips the cancel-and-await dance entirely (the
    ``if not pending.done()`` guard short-circuits) and proceeds.
    Pins the third branch — neither cancellation arm fires when the
    prior drain already finished on its own."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)

    async def already_done() -> None:
        return None

    prior = asyncio.get_running_loop().create_task(already_done())
    # Drain it so it's done before we install it.
    await prior
    assert prior.done()
    conn._pending_drain = prior

    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int):
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
