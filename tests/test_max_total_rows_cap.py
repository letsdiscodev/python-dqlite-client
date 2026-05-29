"""max_total_rows cap actually fires in the continuation-drain loop (mocked stream)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol


def _make_rows_response(rows: list[list[object]], has_more: bool) -> MagicMock:
    r = MagicMock(name="RowsResponse")
    r.rows = rows
    r.has_more = has_more
    r.column_names = ["v"]
    r.column_types = [1]
    r.row_types = [[1] for _ in rows]
    return r


class TestMaxTotalRowsEnforcement:
    def test_initial_frame_with_oversize_rows_raises_before_loop(self) -> None:
        """The cap is cumulative-total: an oversize initial frame raises before any row."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=5)

        # has_more=False so the cap must fire against the initial frame alone.
        initial = _make_rows_response([[i] for i in range(6)], has_more=False)

        # Sentinel: a regression entering the loop would also call _read_continuation.
        called: list[object] = []

        async def fake_read_continuation(deadline: float) -> object:
            called.append(deadline)
            return initial

        p._read_continuation = fake_read_continuation  # type: ignore[assignment]

        async def run() -> None:
            await p._drain_continuations(initial, deadline=999999.0)

        with pytest.raises(ProtocolError, match="max_total_rows"):
            asyncio.run(run())
        assert called == []

    def test_exceeding_cap_raises_protocol_error(self) -> None:
        """A continuation frame that pushes us past max_total_rows raises."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=5)

        initial = _make_rows_response([[1], [2], [3]], has_more=True)
        over_cap = _make_rows_response([[4], [5], [6]], has_more=False)

        p._read_continuation = AsyncMock(return_value=over_cap)

        async def run() -> None:
            await p._drain_continuations(initial, deadline=999999.0)

        with pytest.raises(ProtocolError, match="max_total_rows"):
            asyncio.run(run())

    def test_exactly_at_cap_does_not_raise(self) -> None:
        """Hitting the cap exactly is fine; only exceeding raises."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=5)

        initial = _make_rows_response([[1], [2], [3]], has_more=True)
        at_cap = _make_rows_response([[4], [5]], has_more=False)  # total: 5 == cap

        p._read_continuation = AsyncMock(return_value=at_cap)

        rows, _ = asyncio.run(p._drain_continuations(initial, deadline=999999.0))
        assert len(rows) == 5

    def test_none_disables_cap(self) -> None:
        """max_total_rows=None means the cap never fires."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=None)

        initial = _make_rows_response([[i] for i in range(100)], has_more=True)
        big = _make_rows_response([[i] for i in range(100, 10_000)], has_more=False)

        p._read_continuation = AsyncMock(return_value=big)

        rows, _ = asyncio.run(p._drain_continuations(initial, deadline=999999.0))
        assert len(rows) == 10_000


class TestMaxContinuationFramesEnforcement:
    """Per-frame cap bounds the slow-drip case where many 1-row frames pin a client CPU."""

    def test_exceeding_frame_cap_raises(self) -> None:
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(
            reader,
            writer,
            timeout=5.0,
            max_total_rows=None,
            max_continuation_frames=3,
        )

        initial = _make_rows_response([[1]], has_more=True)
        # Same response each time so the frame counter, not rows, drives termination.
        one_row_continuation = _make_rows_response([[2]], has_more=True)

        p._read_continuation = AsyncMock(return_value=one_row_continuation)

        async def run() -> None:
            await p._drain_continuations(initial, deadline=999999.0)

        with pytest.raises(ProtocolError, match="max_continuation_frames"):
            asyncio.run(run())

    def test_within_frame_cap_succeeds(self) -> None:
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(
            reader,
            writer,
            timeout=5.0,
            max_continuation_frames=5,
        )
        initial = _make_rows_response([[1]], has_more=True)
        last = _make_rows_response([[2], [3]], has_more=False)

        p._read_continuation = AsyncMock(return_value=last)
        rows, _ = asyncio.run(p._drain_continuations(initial, deadline=999999.0))
        assert len(rows) == 3

    def test_none_disables_frame_cap(self) -> None:
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(
            reader,
            writer,
            timeout=5.0,
            max_continuation_frames=None,
        )
        initial = _make_rows_response([[1]], has_more=True)
        last = _make_rows_response([[2]], has_more=False)

        p._read_continuation = AsyncMock(return_value=last)
        rows, _ = asyncio.run(p._drain_continuations(initial, deadline=999999.0))
        assert len(rows) == 2


class TestTrustServerHeartbeat:
    """Server heartbeat no longer widens client timeout by default."""

    def test_default_does_not_amplify_timeout(self) -> None:
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0)
        # Inject the server value directly, bypassing the handshake socket dance.
        p._heartbeat_timeout = 300_000  # 300 s, far above 5 s
        assert p._timeout == 5.0
        assert p._read_timeout == 5.0
        assert p._trust_server_heartbeat is False

    def test_opt_in_respected(self) -> None:
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, trust_server_heartbeat=True)
        assert p._trust_server_heartbeat is True

    def test_opt_in_amplifies_read_timeout_via_handshake(self) -> None:
        """Opt-in widens the per-read deadline but keeps the write-path ``_timeout`` pinned,
        so a long heartbeat advertisement can't stretch writer.drain budgets."""
        from dqlitewire.messages import WelcomeResponse

        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        # heartbeat=60 s: above timeout=5 s, under the 300 s cap
        reader.read.side_effect = [
            WelcomeResponse(heartbeat_timeout=60_000).encode(),
        ]
        p = DqliteProtocol(reader, writer, timeout=5.0, trust_server_heartbeat=True)
        asyncio.run(p.handshake())
        assert p._read_timeout == 60.0, (
            f"trust_server_heartbeat=True should widen read_timeout to 60s, got {p._read_timeout}"
        )
        assert p._timeout == 5.0, (
            f"write-path timeout must stay pinned to the operator value, got {p._timeout}"
        )

    def test_default_ignores_handshake_heartbeat(self) -> None:
        """By default a larger handshake heartbeat is recorded but widens no deadline."""
        from dqlitewire.messages import WelcomeResponse

        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        reader.read.side_effect = [
            WelcomeResponse(heartbeat_timeout=300_000).encode(),
        ]
        p = DqliteProtocol(reader, writer, timeout=5.0)
        asyncio.run(p.handshake())
        # Server value recorded for diagnostics, but both timeouts unchanged.
        assert p._heartbeat_timeout == 300_000
        assert p._timeout == 5.0
        assert p._read_timeout == 5.0

    def test_opt_in_respects_hard_300s_cap(self) -> None:
        """Even opt-in clamps an absurd server heartbeat at 300 s."""
        from dqlitewire.messages import WelcomeResponse

        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        # heartbeat=1 h, clamped to 300 s
        reader.read.side_effect = [
            WelcomeResponse(heartbeat_timeout=3_600_000).encode(),
        ]
        p = DqliteProtocol(reader, writer, timeout=5.0, trust_server_heartbeat=True)
        asyncio.run(p.handshake())
        assert p._read_timeout == 300.0
        assert p._timeout == 5.0
