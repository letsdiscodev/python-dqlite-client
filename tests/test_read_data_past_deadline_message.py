"""Pin: ``_read_data``'s already-past-deadline raise interpolates the
actual observed overrun (``-remaining``), not the per-read budget
``self._read_timeout``.

The prior message reported ``self._read_timeout`` regardless — for a
continuation drain with sub-second remaining, or with
``trust_server_heartbeat=True``-widened ``_read_timeout`` (up to 300s
cap), the value bore no relationship to the configured or the
effective deadline. Diagnostic load-bearing for forensic recovery.
"""

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
    # Configure a widened _read_timeout so any leakage of the field
    # into the message is detectable. The configured operator-level
    # timeout stays at 5s; the read-side window is 300s.
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
    # New format: "exceeded deadline by Ns" with N reflecting the
    # actual overrun budget.
    assert re.search(r"exceeded deadline by \d+\.\d+s", msg), msg
    # Negative regression: the widened _read_timeout MUST NOT appear
    # in the message (its value 300.0 would be the prior misleading
    # interpolation).
    assert "300" not in msg, f"_read_timeout leaked into message; got {msg!r}"
