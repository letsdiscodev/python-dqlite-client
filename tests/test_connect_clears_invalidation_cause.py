"""connect() clears ``_invalidation_cause`` on success so a later silent invalidate
doesn't chain a stale historical failure as ``__cause__``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


@pytest.mark.asyncio
async def test_connect_clears_stale_invalidation_cause(monkeypatch) -> None:
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=1.0)

    stale = RuntimeError("leader flip from earlier cycle")
    conn._invalidation_cause = stale

    fake_protocol = MagicMock()
    fake_protocol.handshake = AsyncMock(return_value=None)
    fake_protocol.open_database = AsyncMock(return_value=7)
    fake_protocol._client_id = 1

    async def _fake_open_connection(host, port, **_kwargs):
        return (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "asyncio.open_connection",
        _fake_open_connection,
    )
    monkeypatch.setattr(
        "dqliteclient.connection.DqliteProtocol",
        lambda *a, **kw: fake_protocol,
    )

    await conn.connect()

    assert conn._invalidation_cause is None

    conn._invalidate(cause=None)
    with pytest.raises(DqliteConnectionError) as exc_info:
        await conn.execute("SELECT 1")
    assert exc_info.value.__cause__ is not stale
