"""Writer drain must respect the per-connection timeout.

Every ``_read_data`` call is wrapped in ``asyncio.wait_for`` with the
per-connection timeout, but ``_send`` awaits ``writer.drain()``
unbounded. A peer that accepts the TCP connection but stops reading
from its socket fills the kernel send buffer and stalls
``drain()`` indefinitely on the high-water-mark future — the
operator-configured timeout never fires on outbound messages. Wrap
the drain the same way reads are wrapped.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.protocol import DqliteProtocol


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=0.1, address="test:9001")


class TestSendDrainTimeout:
    async def test_drain_timeout_raises_dqlite_connection_error(
        self, protocol: DqliteProtocol
    ) -> None:
        """A drain() that never resolves must raise DqliteConnectionError
        within ``self._timeout`` seconds rather than hanging forever.
        """

        async def _never_drain() -> None:
            await asyncio.Event().wait()

        protocol._writer.drain = _never_drain  # type: ignore[attr-defined]

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(DqliteConnectionError, match="timeout|timed out"):
            await protocol._send()
        elapsed = loop.time() - start

        # Timeout is 0.1s; allow a generous slack for scheduling.
        assert elapsed < 1.0, f"drain did not respect timeout; elapsed={elapsed:.3f}s"

    async def test_drain_error_includes_address(self, protocol: DqliteProtocol) -> None:
        """The wrap must preserve the address suffix so operators can tell
        which peer stalled.
        """

        async def _never_drain() -> None:
            await asyncio.Event().wait()

        protocol._writer.drain = _never_drain  # type: ignore[attr-defined]

        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol._send()
        assert "test:9001" in str(exc_info.value)
