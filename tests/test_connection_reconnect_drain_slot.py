"""connect() must clear ``_pending_drain`` on reconnect so a second
``_invalidate`` does not overwrite an un-awaited prior drain task.

Scenario: invalidate → (reconnect path leading back to connect()) →
invalidate overwrites the slot without cancelling the first task,
violating the documented single-ref discipline. Pinning ``connect()``
to cancel-and-await a pending prior drain closes the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_connect_cancels_prior_pending_drain() -> None:
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=1.0)

    # Simulate a still-running drain task scheduled by a prior
    # invalidate.
    ran: list[str] = []

    async def never_completing() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            ran.append("cancelled")
            raise

    prior = asyncio.get_running_loop().create_task(never_completing())
    # Let it actually start running so cancel has something to deliver to.
    await asyncio.sleep(0)
    conn._pending_drain = prior

    # Force connect() through the open_connection path and make it fail
    # promptly so we don't need a real server. We only need to prove
    # connect() processed ``_pending_drain`` before doing its work.
    called: list[object] = []

    async def fake_open_connection(host: str, port: int, **_kwargs: object):
        called.append((host, port))
        # Return a never-read/writer pair via mocks; handshake will
        # be mocked to raise immediately below.
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    # Stub out DqliteProtocol so handshake() raises an
    # OperationalError that does NOT trigger reconnect-worthy branches.
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

    # Either the prior task was cancelled-and-awaited, or the slot was
    # explicitly reset — both satisfy the invariant that a subsequent
    # _invalidate cannot clobber a live task reference.
    assert prior.done(), (
        "connect() must cancel-and-await any prior pending drain so a "
        "subsequent _invalidate does not overwrite a live task reference"
    )
    # Drain the task result so we don't leak a pending task.
    with pytest.raises(asyncio.CancelledError):
        await prior


@pytest.mark.asyncio
async def test_connect_propagates_outer_cancel_during_pending_drain_retire() -> None:
    """An outer ``task.cancel()`` delivered while ``connect()`` is
    awaiting a prior pending-drain task MUST propagate. Without this,
    a broad suppress-after-cancel-and-await would silently consume
    the parent's cancel signal and ``connect()`` would proceed to
    invoke ``asyncio.open_connection`` despite the parent's intent."""
    conn = DqliteConnection("localhost:9001", timeout=5.0, close_timeout=5.0)

    # Long-running drain task standing in for a prior _invalidate's
    # bounded wait. Holds onto execution until externally cancelled.
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_drain() -> None:
        started.set()
        try:
            await proceed.wait()
        except asyncio.CancelledError:
            # Yield before propagating so the outer cancel of driver_task
            # has a chance to land while connect() is awaiting us.
            await asyncio.sleep(0)
            raise

    prior = asyncio.get_running_loop().create_task(slow_drain())
    await started.wait()
    conn._pending_drain = prior

    # Patch open_connection so we can detect whether the cancel signal
    # was honored. If connect() proceeds past the pending-retire, this
    # mock fires and we record the call — that's the regression we are
    # guarding against.
    open_connection_called: list[object] = []
    real_open = asyncio.open_connection

    async def fake_open(host: str, port: int, **_kwargs: object):
        open_connection_called.append((host, port))
        # Hand back enough to survive a first read; never reached on
        # the success path, so an assertion error here is a clear bug.
        return MagicMock(), MagicMock()

    asyncio.open_connection = fake_open  # type: ignore[assignment]
    try:

        async def driver() -> None:
            await conn.connect()

        driver_task = asyncio.get_running_loop().create_task(driver())
        # Let driver enter connect() and reach ``await pending``.
        for _ in range(3):
            await asyncio.sleep(0)

        # Cancel the outer driver task. connect()'s pending-drain await
        # MUST surface the cancel to the next checkpoint.
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
