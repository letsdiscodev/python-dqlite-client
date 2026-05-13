"""Diagnostic logs around the server-advertised heartbeat value.

Three edge cases that the wire layer accepts but cannot remediate
client-side. Each is non-fatal — but the operator chasing a
mis-configured-peer / non-conforming-server symptom needs the
breadcrumb to correlate against per-cluster config audits.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import WelcomeResponse


@pytest.fixture
def mock_reader() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_writer() -> MagicMock:
    w = MagicMock()
    w.write = MagicMock(return_value=None)
    w.drain = AsyncMock(return_value=None)
    w.is_closing = MagicMock(return_value=False)
    return w


@pytest.mark.asyncio
async def test_zero_heartbeat_emits_diagnostic_debug(
    mock_reader: AsyncMock,
    mock_writer: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A server advertising ``heartbeat_timeout=0`` is flagged at DEBUG
    as semantically ambiguous (per the wire spec)."""
    mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=0).encode()
    protocol = DqliteProtocol(mock_reader, mock_writer, timeout=5.0)
    caplog.set_level(logging.DEBUG, logger="dqliteclient.protocol")
    await protocol.handshake()
    messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.protocol"]
    assert any("advertised heartbeat=0" in m and "ambiguous" in m for m in messages), (
        f"expected 'heartbeat=0' diagnostic DEBUG; got {messages!r}"
    )


@pytest.mark.asyncio
async def test_over_cap_heartbeat_emits_clip_warning(
    mock_reader: AsyncMock,
    mock_writer: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A server advertising a heartbeat over the 300 s client cap emits
    a WARNING so the operator can tell the value was clipped."""
    # 10000 seconds -> capped to 300 s.
    mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=10_000_000).encode()
    protocol = DqliteProtocol(mock_reader, mock_writer, timeout=5.0, trust_server_heartbeat=True)
    caplog.set_level(logging.DEBUG, logger="dqliteclient.protocol")
    await protocol.handshake()
    warn_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dqliteclient.protocol"
    ]
    assert warn_records, "expected a WARNING about heartbeat cap clipping"
    msg = warn_records[0].getMessage()
    assert "exceeds" in msg and "clipping" in msg


@pytest.mark.asyncio
async def test_heartbeat_smaller_than_read_timeout_emits_no_op_debug(
    mock_reader: AsyncMock,
    mock_writer: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``trust_server_heartbeat=True`` plus a server-advertised heartbeat
    smaller than the operator's configured read deadline emits a DEBUG
    explaining the no-op widening. Operator opted in expecting
    widening; the contradiction needs a breadcrumb."""
    # 1000 ms = 1 s; operator configured 5 s read deadline.
    mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=1_000).encode()
    protocol = DqliteProtocol(mock_reader, mock_writer, timeout=5.0, trust_server_heartbeat=True)
    caplog.set_level(logging.DEBUG, logger="dqliteclient.protocol")
    await protocol.handshake()
    messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.protocol"]
    assert any("no widening applied" in m for m in messages), (
        f"expected 'no widening applied' DEBUG; got {messages!r}"
    )
