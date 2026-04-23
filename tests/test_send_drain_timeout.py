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


class TestSendDrainMessageShape:
    """``_send`` must emit distinct message shapes for the
    ``TimeoutError`` arm ("Write timeout ... after Xs") vs the
    ``OSError`` / ``RuntimeError`` family ("Write failed: ...").

    SQLAlchemy's ``is_disconnect`` substring scan keys on both
    shapes; collapsing them into one would lose the diagnostic
    distinction between a wedged peer (timeout) and an actively
    closed transport (reset / broken pipe).
    """

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
            await protocol._send()

        msg = str(exc_info.value)
        assert expected_substr in msg, (
            f"expected {expected_substr!r} in {msg!r}; the arm should "
            f"differentiate from 'Write timeout'"
        )
        # Address suffix is part of the contract for both arms.
        assert "peer:9001" in msg
        # __cause__ is preserved so the underlying class is still
        # inspectable (used by SA-side retries).
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
            await protocol._send()
        msg = str(exc_info.value)
        assert "Write timeout" in msg, (
            f"expected 'Write timeout' in {msg!r}; the timeout arm must "
            f"NOT collapse into the 'Write failed' shape"
        )
        assert "Write failed" not in msg
        assert "peer:9001" in msg
