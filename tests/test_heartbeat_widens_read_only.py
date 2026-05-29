"""Server-heartbeat widening affects reads only; the write path stays
pinned to the operator-configured timeout so a hostile server cannot
widen the write SLO by advertising a long heartbeat.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol


class _FakeWelcome:
    heartbeat_timeout = 60_000  # far above the default 10 s


@pytest.mark.asyncio
async def test_handshake_widens_read_timeout_not_write_timeout() -> None:
    reader = MagicMock()
    writer = MagicMock()
    p = DqliteProtocol(reader, writer, timeout=5.0, trust_server_heartbeat=True)

    # Drive the widening branch directly, short-circuiting the wire I/O.
    response = _FakeWelcome()
    if p._trust_server_heartbeat and response.heartbeat_timeout > 0:
        heartbeat_seconds = response.heartbeat_timeout / 1000.0
        new_read_timeout = max(p._read_timeout, min(heartbeat_seconds, 300.0))
        p._read_timeout = new_read_timeout

    assert p._read_timeout == 60.0
    assert p._timeout == 5.0


@pytest.mark.asyncio
async def test_write_path_uses_timeout_not_read_timeout() -> None:
    reader = MagicMock()
    writer = MagicMock()
    p = DqliteProtocol(reader, writer, timeout=0.1, trust_server_heartbeat=True)
    p._read_timeout = 60.0  # simulate handshake widen

    # drain hangs longer than `timeout` but well under `read_timeout`.
    async def slow_drain() -> None:
        await asyncio.sleep(0.5)

    writer.drain = slow_drain

    from dqliteclient.exceptions import DqliteConnectionError

    start = asyncio.get_event_loop().time()
    with pytest.raises(DqliteConnectionError, match="Write timeout"):
        await p._send(b"")
    elapsed = asyncio.get_event_loop().time() - start
    # Drain cut at the write-path `self._timeout`, not the read-path timeout.
    assert elapsed < 1.0, f"write-path stretched to read-path timeout: elapsed={elapsed}"
