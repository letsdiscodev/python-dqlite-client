"""``close()`` → ``connect()`` on the same connection must clear the sticky ``_closed`` /
``_closed_flag[0]`` markers. Otherwise ``conn.closed`` reads True on a working reconnected
slot, and the GC-time ``_connection_unclosed_warning`` finalizer short-circuits, masking a
reconnected-then-leaked socket."""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


async def test_closed_property_reflects_state_after_simulated_reconnect() -> None:
    """After a stubbed reconnect, both ``closed`` and ``_closed_flag[0]`` must be False."""
    conn = DqliteConnection("localhost:9001")

    # Seed a closed state mirroring what close() leaves behind.
    conn._closed = True
    conn._closed_flag[0] = True
    conn._protocol = None
    conn._db_id = None

    # Patch success-path internals so _connect_impl proceeds without a live cluster;
    # the flag-clearing site lives in the success branch after open_database.
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_protocol = MagicMock()
    fake_protocol.handshake = AsyncMock(return_value=None)
    fake_protocol.open_database = AsyncMock(return_value=42)
    fake_protocol.close = MagicMock()

    async def _fake_open(*args: object, **kwargs: object) -> object:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock(return_value=None)
        return reader, writer

    with (
        patch("dqliteclient.connection.open_connection", new=_fake_open),
        patch("dqliteclient.connection.DqliteProtocol", return_value=fake_protocol),
    ):
        await conn.connect()

    assert conn.closed is False, (
        f"close() → connect() must reset .closed to False; got closed={conn.closed!r}"
    )
    assert conn._closed_flag[0] is False, (
        "close() → connect() must reset _closed_flag[0] so the "
        f"GC-time ResourceWarning re-arms; got {conn._closed_flag[0]!r}"
    )
    assert conn._protocol is fake_protocol
    assert conn._db_id == 42
