"""connect() must clear ``_invalidation_cause`` on a successful
reconnect so a subsequent silent ``_invalidate()`` (cause=None) does not
surface "Not connected" errors whose ``__cause__`` chain points at an
unrelated historical failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


@pytest.mark.asyncio
async def test_connect_clears_stale_invalidation_cause(monkeypatch) -> None:
    """After a prior ``_invalidate(cause=ServerFailure)`` followed by a
    successful ``connect()``, a subsequent silent invalidation must not
    chain the stale cause."""
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=1.0)

    # Seed a stale cause as if a prior failure invalidated the conn.
    stale = RuntimeError("leader flip from earlier cycle")
    conn._invalidation_cause = stale

    # Patch out the networking: make open_connection/handshake/open_database
    # all succeed so connect() reaches the end of the happy path.
    fake_protocol = MagicMock()
    fake_protocol.handshake = AsyncMock(return_value=None)
    fake_protocol.open_database = AsyncMock(return_value=7)
    fake_protocol._client_id = 1

    async def _fake_open_connection(host, port):
        return (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "dqliteclient.connection.asyncio.open_connection",
        _fake_open_connection,
    )
    monkeypatch.setattr(
        "dqliteclient.connection.DqliteProtocol",
        lambda *a, **kw: fake_protocol,
    )

    await conn.connect()

    # Successful reconnect must have cleared the stale cause.
    assert conn._invalidation_cause is None

    # Simulate a later silent invalidate and confirm the resulting
    # "Not connected" error has no stale chain.
    conn._invalidate(cause=None)
    with pytest.raises(DqliteConnectionError) as exc_info:
        await conn.execute("SELECT 1")
    # The "Not connected" error should NOT chain the old stale cause.
    assert exc_info.value.__cause__ is not stale
