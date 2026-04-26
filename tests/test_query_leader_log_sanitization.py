"""Pin: ``_query_leader``'s DEBUG log scrubs control / ANSI / bidi
bytes from the server-supplied ``leader_addr``.

The exception message that follows already routes ``leader_addr``
through ``_sanitize_display_text``; the DEBUG log used to emit it
verbatim via ``%r``. A hostile (or compromised) peer can return a
crafted address — embedded ``\\r\\n``, ANSI escape sequences, bidi
override glyphs — that injects content into the operator's log
stream. DEBUG logs are routinely shipped to aggregators; defence in
depth means scrubbing both surfaces.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ProtocolError
from dqliteclient.node_store import MemoryNodeStore

_HOSTILE_ADDR = "evil:9001\r\n2026-01-01 ERROR FAKE LEADER ELECTED \x1b[31m"
_BIDI_ADDR = "node‮evil:9001"


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


@pytest.mark.asyncio
async def test_query_leader_empty_addr_with_nonzero_id_sanitizes_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cluster = _make_cluster()

    async def fake_open_connection(host: str, port: int) -> tuple[MagicMock, MagicMock]:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock(return_value=10_000)
    # node_id != 0 with empty address triggers the first malformed-redirect arm.
    fake_proto.get_leader = AsyncMock(return_value=(7, ""))

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.cluster"),
        pytest.raises(ProtocolError),
    ):
        # Patch open_connection + protocol so _query_leader exercises
        # the malformed-redirect path without touching the network.
        from unittest.mock import patch

        with (
            patch("dqliteclient.cluster.asyncio.open_connection", new=fake_open_connection),
            patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        ):
            await cluster._query_leader("localhost:9001", trust_server_heartbeat=False)


def _assert_no_raw_control_bytes(records: list[logging.LogRecord]) -> None:
    for r in records:
        msg = r.getMessage()
        assert "\r" not in msg, f"raw CR in {msg!r}"
        assert "\n" not in msg or r.levelno >= logging.WARNING, f"raw LF in DEBUG {msg!r}"
        assert "\x1b" not in msg, f"raw ANSI escape in {msg!r}"
        assert "‮" not in msg, f"raw bidi override in {msg!r}"


@pytest.mark.asyncio
async def test_query_leader_zero_id_with_hostile_addr_sanitizes_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cluster = _make_cluster()

    async def fake_open_connection(host: str, port: int) -> tuple[MagicMock, MagicMock]:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock(return_value=10_000)
    # node_id == 0 with non-empty hostile address triggers the second arm.
    fake_proto.get_leader = AsyncMock(return_value=(0, _HOSTILE_ADDR))

    from unittest.mock import patch

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.cluster"),
        patch("dqliteclient.cluster.asyncio.open_connection", new=fake_open_connection),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(ProtocolError),
    ):
        await cluster._query_leader("localhost:9001", trust_server_heartbeat=False)

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "expected DEBUG records from _query_leader"
    _assert_no_raw_control_bytes(debug_records)


@pytest.mark.asyncio
async def test_query_leader_bidi_addr_in_debug_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cluster = _make_cluster()

    async def fake_open_connection(host: str, port: int) -> tuple[MagicMock, MagicMock]:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock(return_value=10_000)
    fake_proto.get_leader = AsyncMock(return_value=(0, _BIDI_ADDR))

    from unittest.mock import patch

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.cluster"),
        patch("dqliteclient.cluster.asyncio.open_connection", new=fake_open_connection),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(ProtocolError),
    ):
        await cluster._query_leader("localhost:9001", trust_server_heartbeat=False)

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    _assert_no_raw_control_bytes(debug_records)
