"""``_send`` must wrap ``writer.drain()`` in the per-connection timeout: a peer that
accepts the TCP connection but stops reading fills the send buffer and stalls drain()
forever on the high-water-mark future, so an unbounded drain ignores the timeout."""

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
        """A drain() that never resolves must raise DqliteConnectionError, not hang."""

        async def _never_drain() -> None:
            await asyncio.Event().wait()

        protocol._writer.drain = _never_drain

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(DqliteConnectionError, match="timeout|timed out"):
            await protocol._send(b"")
        elapsed = loop.time() - start

        assert elapsed < 1.0, f"drain did not respect timeout; elapsed={elapsed:.3f}s"

    async def test_drain_error_includes_address(self, protocol: DqliteProtocol) -> None:
        """The error must include the address suffix so operators can tell which peer stalled."""

        async def _never_drain() -> None:
            await asyncio.Event().wait()

        protocol._writer.drain = _never_drain

        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol._send(b"")
        assert "test:9001" in str(exc_info.value)


class TestSendDrainMessageShape:
    """``_send`` emits distinct shapes ("Write timeout" vs "Write failed") that SA's
    ``is_disconnect`` keys on, distinguishing a wedged peer from a closed transport."""

    @pytest.mark.parametrize(
        "raised,expected_substr",
        [
            pytest.param(BrokenPipeError("pipe broken"), "Write failed", id="broken-pipe"),
            pytest.param(
                ConnectionResetError("reset by peer"), "Write failed", id="connection-reset"
            ),
            pytest.param(OSError("ENOTCONN"), "Write failed", id="generic-oserror"),
            pytest.param(RuntimeError("Transport is closed"), "Write failed", id="runtime-error"),
        ],
    )
    async def test_oserror_family_raises_write_failed(
        self, raised: BaseException, expected_substr: str
    ) -> None:
        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock(side_effect=raised)
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        protocol = DqliteProtocol(reader, writer, timeout=0.5, address="peer:9001")

        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol._send(b"")

        msg = str(exc_info.value)
        assert expected_substr in msg, (
            f"expected {expected_substr!r} in {msg!r}; the arm should "
            f"differentiate from 'Write timeout'"
        )
        assert "peer:9001" in msg
        assert exc_info.value.__cause__ is raised

    async def test_timeout_error_raises_write_timeout(self) -> None:
        reader = AsyncMock()
        writer = MagicMock()

        async def _never_drain() -> None:
            await asyncio.Event().wait()

        writer.drain = _never_drain
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        protocol = DqliteProtocol(reader, writer, timeout=0.05, address="peer:9001")

        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol._send(b"")
        msg = str(exc_info.value)
        assert "Write timeout" in msg, (
            f"expected 'Write timeout' in {msg!r}; the timeout arm must "
            f"NOT collapse into the 'Write failed' shape"
        )
        assert "Write failed" not in msg
        assert "peer:9001" in msg
