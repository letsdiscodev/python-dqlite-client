"""Pin: ``_read_data``'s past-deadline raise reports the actual observed overrun
(``-remaining``), not the per-read budget ``self._read_timeout`` (which can be
widened to 300s under ``trust_server_heartbeat`` and bear no relation to the deadline)."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.protocol import DqliteProtocol


def _make_proto() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    # Widen _read_timeout (300s) away from the 5s operator timeout so leakage is detectable.
    proto = DqliteProtocol(reader, writer, timeout=5.0)
    proto._read_timeout = 300.0
    return proto


@pytest.mark.asyncio
async def test_past_deadline_message_reports_overrun_not_read_timeout() -> None:
    proto = _make_proto()
    loop = asyncio.get_running_loop()
    past_deadline = loop.time() - 1.5  # ~1.5s past

    with pytest.raises(DqliteConnectionError) as ei:
        await proto._read_data(deadline=past_deadline)

    msg = str(ei.value)
    assert re.search(r"exceeded deadline by \d+\.\d+s", msg), msg
    # The widened _read_timeout (300.0) must not leak into the message.
    assert "300" not in msg, f"_read_timeout leaked into message; got {msg!r}"
