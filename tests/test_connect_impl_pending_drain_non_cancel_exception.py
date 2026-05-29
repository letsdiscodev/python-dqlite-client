"""``_connect_impl``'s ``except Exception: pass`` arm swallows a non-Cancel
Exception from awaiting a prior ``_pending_drain`` task so reconnect proceeds."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import dqliteclient.connection as conn_mod
from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_connect_swallows_non_cancel_exception_from_prior_drain() -> None:
    """The caller sees the fresh attempt's dial-failure, not the swallowed
    drain RuntimeError."""
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=1.0)

    # Production shape: an ``_invalidate`` whose bounded ``wait_closed`` fails
    # at the transport layer re-raises RuntimeError when cancelled.
    started = asyncio.Event()

    async def drain_raising_runtime() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise RuntimeError("synthetic drain transport failure") from None

    prior = asyncio.get_running_loop().create_task(drain_raising_runtime())
    await started.wait()
    conn._pending_drain = prior

    # Dial fails with a distinguishable error so we can assert the user sees it,
    # not the drain's RuntimeError.
    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_open = asyncio.open_connection

    async def fake_open_connection(host: str, port: int, **_kwargs: object):
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    class _StubProtocol:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._client_id = 0
            self._writer = args[1] if len(args) >= 2 else MagicMock()

        async def handshake(self) -> None:
            raise RuntimeError("dial-failure-marker")

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
        with pytest.raises(RuntimeError, match="dial-failure-marker"):
            await conn.connect()
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        asyncio.open_connection = real_open
        DqliteConnection._abort_protocol = original_abort

    assert conn._pending_drain is None
    assert prior.done()
    # Confirm it raised the non-Cancel Exception rather than just being cancelled.
    with pytest.raises(RuntimeError, match="synthetic drain transport failure"):
        await prior
