"""The half-constructed-protocol cleanup drain in connect() honours self._close_timeout."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_connect_cleanup_uses_configured_close_timeout() -> None:
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=2.0)

    fake_reader = MagicMock()
    fake_writer = MagicMock()
    fake_writer.close = MagicMock()
    fake_writer.wait_closed = AsyncMock()

    async def fake_open_connection(host: str, port: int, **_kwargs: object):
        return fake_reader, fake_writer

    import dqliteclient.connection as conn_mod

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_open = asyncio.open_connection
    real_timeout = asyncio.timeout

    def broken_protocol_init(*args: object, **kwargs: object) -> None:
        raise RuntimeError("protocol construction blew up")

    class _BrokenProtocol:
        def __init__(self, *a: object, **kw: object) -> None:
            broken_protocol_init()

    captured_timeouts: list[float | None] = []

    def spy_timeout(timeout: float | None):
        captured_timeouts.append(timeout)
        return real_timeout(timeout)

    conn_mod.DqliteProtocol = _BrokenProtocol  # type: ignore[assignment,attr-defined]
    asyncio.open_connection = fake_open_connection  # type: ignore[assignment]
    conn_mod.asyncio.timeout = spy_timeout  # type: ignore[attr-defined]

    try:
        with pytest.raises(RuntimeError, match="protocol construction"):
            await conn.connect()
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        asyncio.open_connection = real_open
        conn_mod.asyncio.timeout = real_timeout  # type: ignore[attr-defined]

    assert captured_timeouts, "expected asyncio.timeout to run during cleanup"
    assert 2.0 in captured_timeouts, (
        "protocol-construction cleanup must honour self._close_timeout; "
        f"got timeouts {captured_timeouts}"
    )
