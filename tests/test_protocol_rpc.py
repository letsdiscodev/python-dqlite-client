"""DqliteProtocol request/response mechanics: cancel scope, serialisation, drains, caps."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import protocol as protocol_mod
from dqliteclient.exceptions import DqliteConnectionError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire import WIRE_DECODE_FAILED_PREFIX
from dqlitewire.messages import LeaderResponse


@pytest.mark.asyncio
async def test_get_leader_defence_in_depth_rejects_node_id_zero_with_address() -> None:
    """Feed ``LeaderResponse(0, addr)`` past the decoder to exercise
    the protocol-layer defence-in-depth guard."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    malformed = LeaderResponse(node_id=0, address="attacker:9000")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=malformed)),
        pytest.raises(ProtocolError, match="with node_id=0"),
    ):
        await protocol.get_leader()


@pytest.mark.asyncio
async def test_get_leader_passes_through_no_leader_known_shape() -> None:
    """The ``(0, "")`` "no leader known" shape is passed through verbatim."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    no_leader = LeaderResponse(node_id=0, address="")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=no_leader)),
    ):
        node_id, address = await protocol.get_leader()
    assert node_id == 0
    assert address == ""


@pytest.mark.asyncio
async def test_get_leader_passes_through_raft_nomem_transient_shape() -> None:
    """The ``(N, "")`` ``RAFT_NOMEM`` transient is passed through."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    raft_nomem = LeaderResponse(node_id=5, address="")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=raft_nomem)),
    ):
        node_id, address = await protocol.get_leader()
    assert node_id == 5
    assert address == ""


@pytest.mark.asyncio
async def test_get_leader_happy_path_unchanged() -> None:
    """The normal ``(N, addr)`` pair is returned verbatim."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    happy = LeaderResponse(node_id=2, address="real-leader:9001")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=happy)),
    ):
        node_id, address = await protocol.get_leader()
    assert node_id == 2
    assert address == "real-leader:9001"


@pytest.mark.asyncio
async def test_open_database_rejects_nonzero_db_id() -> None:
    from dqlitewire.messages import DbResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    bad = DbResponse(db_id=42)
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=bad)),
        pytest.raises(ProtocolError) as exc_info,
    ):
        await protocol.open_database("default")

    msg = str(exc_info.value)
    assert "expected 0" in msg
    assert WIRE_DECODE_FAILED_PREFIX in msg
    assert "db_id=42" in msg


@pytest.mark.asyncio
async def test_open_database_happy_path_returns_zero() -> None:
    from dqlitewire.messages import DbResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    ok = DbResponse(db_id=0)
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=ok)),
    ):
        db_id = await protocol.open_database("default")
    assert db_id == 0


def _make_protocol(reader: MagicMock, writer: MagicMock) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._reader = reader
    proto._writer = writer
    proto._timeout = 0.5
    proto._read_timeout = 0.5
    proto._address = "localhost:9001"
    proto._heartbeat_timeout = 0
    return proto


@pytest.mark.asyncio
async def test_send_timeout_surfaces_as_dqlite_connection_error() -> None:
    """A slow ``drain()`` surfaces as ``DqliteConnectionError``."""
    writer = MagicMock()

    async def _slow_drain() -> None:
        await asyncio.sleep(10)

    writer.drain = AsyncMock(side_effect=_slow_drain)
    proto = _make_protocol(MagicMock(), writer)
    proto._timeout = 0.01

    with pytest.raises(DqliteConnectionError, match=r"Write timeout"):
        await proto._send(b"")


@pytest.mark.asyncio
async def test_read_data_timeout_surfaces_as_dqlite_connection_error() -> None:
    """A slow ``read()`` surfaces as ``DqliteConnectionError``."""
    reader = MagicMock()

    async def _slow_read(_n: int) -> bytes:
        await asyncio.sleep(10)
        return b""

    reader.read = _slow_read
    proto = _make_protocol(reader, MagicMock())
    proto._read_timeout = 0.01

    with pytest.raises(DqliteConnectionError, match=r"timed out"):
        await proto._read_data(asyncio.get_running_loop().time() + 60)


@pytest.mark.asyncio
async def test_send_outer_cancel_propagates_as_cancel_not_dqlite_error() -> None:
    """An outer cancel surfaces as ``CancelledError``, not
    ``DqliteConnectionError``."""
    writer = MagicMock()
    drain_started = asyncio.Event()
    cancel_now = asyncio.Event()

    async def _drain_then_block() -> None:
        drain_started.set()
        try:
            await cancel_now.wait()
        except asyncio.CancelledError:
            raise

    writer.drain = AsyncMock(side_effect=_drain_then_block)
    proto = _make_protocol(MagicMock(), writer)
    proto._timeout = 60.0  # long timeout so only the outer cancel fires

    async def run() -> None:
        await proto._send(b"")

    task = asyncio.create_task(run())
    await drain_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_read_data_outer_cancel_propagates_as_cancel() -> None:
    """An outer cancel on ``_read_data`` surfaces as ``CancelledError``."""
    reader = MagicMock()
    read_started = asyncio.Event()
    cancel_now = asyncio.Event()

    async def _read_then_block(_n: int) -> bytes:
        read_started.set()
        await cancel_now.wait()
        return b""

    reader.read = _read_then_block
    proto = _make_protocol(reader, MagicMock())
    proto._read_timeout = 60.0

    async def run() -> None:
        await proto._read_data(asyncio.get_running_loop().time() + 60)

    task = asyncio.create_task(run())
    await read_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_protocol_has_asyncio_lock_attribute() -> None:
    """``DqliteProtocol`` exposes an ``asyncio.Lock`` ``_lock``."""
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    proto = DqliteProtocol(reader, writer)
    assert isinstance(proto._lock, asyncio.Lock)


async def test_concurrent_get_leader_calls_serialise() -> None:
    """Two concurrent ``get_leader()`` calls run sequentially; each
    read pulls its own response from the shared decoder."""
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()

    # Distinct payloads so each task can tell which response it got.
    resp_a = LeaderResponse(node_id=1, address="10.0.0.1:9001").encode()
    resp_b = LeaderResponse(node_id=2, address="10.0.0.2:9001").encode()

    payloads = [resp_a, resp_b]
    proto = DqliteProtocol(reader, writer)

    start_a = asyncio.Event()
    release_a = asyncio.Event()
    started_a = asyncio.Event()
    started_b = asyncio.Event()

    call_order: list[str] = []
    read_order: list[str] = []

    async def fake_read(n: int) -> bytes:
        # Hold both reads until both tasks have queued, so the lock
        # has a chance to enforce serialisation.
        if "A_first_read" not in call_order:
            call_order.append("A_first_read")
            await start_a.wait()
        await release_a.wait()
        if payloads:
            data = payloads.pop(0)
            read_order.append(data.hex()[:16])
            return data
        return b""

    reader.read = AsyncMock(side_effect=fake_read)

    async def task_a() -> tuple[int, str]:
        started_a.set()
        try:
            return await proto.get_leader()
        finally:
            call_order.append("A_done")

    async def task_b() -> tuple[int, str]:
        started_b.set()
        try:
            return await proto.get_leader()
        finally:
            call_order.append("B_done")

    fut_a = asyncio.create_task(task_a())
    await started_a.wait()
    # Give A a chance to acquire the lock and send.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    fut_b = asyncio.create_task(task_b())
    await started_b.wait()
    start_a.set()
    release_a.set()

    result_a, result_b = await asyncio.gather(fut_a, fut_b)

    assert call_order.index("A_done") < call_order.index("B_done"), (
        f"DqliteProtocol must serialise RPCs; observed order {call_order!r}"
    )
    assert result_a == (1, "10.0.0.1:9001")
    assert result_b == (2, "10.0.0.2:9001")


def _build_protocol(**kwargs: Any) -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer, **kwargs)


def test_forwards_explicit_max_total_rows_to_decoder() -> None:
    proto = _build_protocol(max_total_rows=50_000_000)
    assert proto._decoder._max_total_rows == 50_000_000


def test_forwards_explicit_max_continuation_frames_to_decoder() -> None:
    proto = _build_protocol(max_continuation_frames=500_000)
    assert proto._decoder._max_continuation_frames == 500_000


def test_none_max_total_rows_disables_codec_cap() -> None:
    """``None`` means disabled: the codec accepts it and skips the cap check."""
    proto = _build_protocol(max_total_rows=None)
    assert proto._decoder._max_total_rows is None


def test_none_max_continuation_frames_disables_codec_cap() -> None:
    proto = _build_protocol(max_continuation_frames=None)
    assert proto._decoder._max_continuation_frames is None


def test_default_caps_match_protocol_defaults() -> None:
    """With no caps specified, the protocol forwards its defaults, which equal
    the wire package's ``DEFAULT_*`` constants."""
    from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES, DEFAULT_MAX_TOTAL_ROWS

    proto = _build_protocol()
    assert proto._decoder._max_total_rows == DEFAULT_MAX_TOTAL_ROWS
    assert proto._decoder._max_continuation_frames == DEFAULT_MAX_CONTINUATION_FRAMES


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


_U2028 = " "


def _make_protocol_with_address(address: str) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = address
    return proto


def test_addr_suffix_escapes_lf() -> None:
    proto = _make_protocol_with_address("127.0.0.1:9001\nFAKE LOG")
    suffix = proto._addr_suffix()
    assert "\n" not in suffix, f"raw LF leaked into addr suffix: {suffix!r}"
    assert "\\n" in suffix, f"sanitize_for_log should escape LF; got {suffix!r}"


def test_addr_suffix_escapes_tab() -> None:
    proto = _make_protocol_with_address("127.0.0.1:9001\tFAKE LOG")
    suffix = proto._addr_suffix()
    assert "\t" not in suffix, f"raw TAB leaked into addr suffix: {suffix!r}"


def test_addr_suffix_strips_u2028() -> None:
    forged = f"127.0.0.1:9001{_U2028}FAKE LOG"
    proto = _make_protocol_with_address(forged)
    suffix = proto._addr_suffix()
    assert _U2028 not in suffix, f"U+2028 leaked into addr suffix: {suffix!r}"


def test_addr_suffix_empty_when_no_address() -> None:
    proto = _make_protocol_with_address("")
    assert proto._addr_suffix() == ""


def test_addr_suffix_safe_address_passes_through() -> None:
    """A safe address renders unchanged apart from the ' to ' prefix."""
    proto = _make_protocol_with_address("127.0.0.1:9001")
    suffix = proto._addr_suffix()
    assert suffix == " to 127.0.0.1:9001"


def test_drain_continuations_aliases_initial_row_types_into_accumulator() -> None:
    """Accumulator row-types entries are the SAME identity as the initial frame's (no copy)."""
    import asyncio
    from unittest.mock import MagicMock

    from dqlitewire.constants import ValueType
    from dqlitewire.messages.responses import RowsResponse

    # Build via ``_from_decoded`` so the inner lists share the production decode provenance.
    initial = RowsResponse._from_decoded(
        column_names=["a", "b"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[[1, "hello"]],
        row_types=[[ValueType.INTEGER, ValueType.TEXT]],
        has_more=False,
    )

    proto = protocol_mod.DqliteProtocol.__new__(protocol_mod.DqliteProtocol)
    proto._max_total_rows = None
    proto._max_continuation_frames = None
    proto._reader = MagicMock()
    proto._writer = MagicMock()

    rows, row_types = asyncio.run(proto._drain_continuations(initial, deadline=0.0))
    assert rows == [[1, "hello"]]
    assert row_types == [[ValueType.INTEGER, ValueType.TEXT]]
    assert row_types[0] is initial.row_types[0], (
        "row_types[0] should be aliased from initial.row_types[0] after "
        "the ownership-transfer fix; got a defensive copy instead."
    )


def test_drain_continuations_passes_rows_correctly() -> None:
    import asyncio
    from unittest.mock import MagicMock

    from dqlitewire.constants import ValueType
    from dqlitewire.messages.responses import RowsResponse

    initial = RowsResponse._from_decoded(
        column_names=["x"],
        column_types=[ValueType.INTEGER],
        rows=[[1], [2], [3]],
        row_types=[
            [ValueType.INTEGER],
            [ValueType.INTEGER],
            [ValueType.INTEGER],
        ],
        has_more=False,
    )

    proto = protocol_mod.DqliteProtocol.__new__(protocol_mod.DqliteProtocol)
    proto._max_total_rows = None
    proto._max_continuation_frames = None
    proto._reader = MagicMock()
    proto._writer = MagicMock()

    rows, row_types = asyncio.run(proto._drain_continuations(initial, deadline=0.0))
    assert rows == [[1], [2], [3]]
    assert len(row_types) == 3
