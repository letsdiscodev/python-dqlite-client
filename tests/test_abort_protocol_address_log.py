"""Pin the address in _abort_protocol's unexpected-drain DEBUG line.

Every other DEBUG line in the connection module includes
``self._address`` so operators can correlate events across a flapping
node. The unexpected-drain branch was the one exception; fence it.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_abort_protocol_logs_address_on_unexpected_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn = DqliteConnection("node-7:9000")

    # Stand up a mock protocol whose wait_closed raises an
    # unexpected-category error.
    protocol = MagicMock()
    protocol.close = MagicMock()
    protocol.wait_closed = AsyncMock(side_effect=ValueError("unexpected"))
    conn._protocol = protocol

    with caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"):
        await conn._abort_protocol()

    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and "_abort_protocol" in r.getMessage()
        and "node-7:9000" in r.getMessage()
    ]
    assert matching, f"expected DEBUG '_abort_protocol' record with address; got {caplog.records!r}"


@pytest.mark.asyncio
async def test_abort_protocol_timeout_stays_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expected-category exceptions (TimeoutError, OSError) stay quiet —
    the DEBUG line is reserved for truly unexpected drain errors.
    """
    conn = DqliteConnection("node-7:9000")

    async def _slow_forever() -> None:
        await asyncio.sleep(10)

    protocol = MagicMock()
    protocol.close = MagicMock()
    protocol.wait_closed = AsyncMock(side_effect=_slow_forever)
    conn._protocol = protocol

    with caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"):
        await conn._abort_protocol()

    # No "_abort_protocol: unexpected drain error" record.
    assert not any(
        "_abort_protocol: unexpected drain error" in r.getMessage() for r in caplog.records
    )
