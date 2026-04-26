"""connect() must clear ``_pending_drain`` on reconnect so a second
``_invalidate`` does not overwrite an un-awaited prior drain task.

Scenario: invalidate → (reconnect path leading back to connect()) →
invalidate overwrites the slot without cancelling the first task,
violating the documented single-ref discipline. Pinning ``connect()``
to cancel-and-await a pending prior drain closes the loop.
"""

from __future__ import annotations

import asyncio
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

    async def fake_open_connection(host: str, port: int):
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
