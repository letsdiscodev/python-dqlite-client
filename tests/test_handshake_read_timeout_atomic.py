"""Pin: ``handshake`` either widens ``_read_timeout`` atomically (when
opted in) or leaves the operator-configured value unchanged.

Two contracts:

1. ``trust_server_heartbeat=False`` (default): the server-advertised
   heartbeat is recorded for diagnostics but MUST NOT widen
   ``_read_timeout``. A regression that always widens would let a
   hostile server stretch the operator's read SLO up to 300 s.
2. ``trust_server_heartbeat=True``: widens up to the
   ``_HEARTBEAT_READ_TIMEOUT_CAP_SECONDS`` cap and never narrows.

The widen is the very last write before ``handshake`` returns, so
torn-state is impossible — but the contract still has to hold under
extreme advertised values (negative, zero, huge).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS, DqliteProtocol
from dqlitewire.messages import WelcomeResponse


def _make_protocol(timeout: float, *, trust: bool) -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=timeout, trust_server_heartbeat=trust)


@pytest.mark.asyncio
async def test_default_handshake_does_not_widen_read_timeout() -> None:
    """Pin: with ``trust_server_heartbeat=False`` (default), a server
    advertising a 300 s heartbeat does NOT widen the operator's
    configured per-read deadline."""
    proto = _make_protocol(timeout=5.0, trust=False)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=300_000))

    await proto.handshake()

    assert proto._read_timeout == 5.0
    # Heartbeat is recorded for diagnostics either way.
    assert proto._heartbeat_timeout == 300_000


@pytest.mark.asyncio
async def test_trust_handshake_widens_up_to_cap() -> None:
    """Pin: with ``trust=True`` and a server heartbeat above the cap,
    ``_read_timeout`` widens to the cap and not beyond."""
    proto = _make_protocol(timeout=5.0, trust=True)
    proto._send = AsyncMock()
    # Advertise 600 s; cap is 300.
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=600_000))

    await proto.handshake()

    assert proto._read_timeout == _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS


@pytest.mark.asyncio
async def test_trust_handshake_does_not_narrow_read_timeout() -> None:
    """Pin: a small server heartbeat (e.g., 1 s) does not narrow the
    operator's larger configured timeout."""
    proto = _make_protocol(timeout=30.0, trust=True)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=1_000))

    await proto.handshake()

    assert proto._read_timeout == 30.0


@pytest.mark.asyncio
async def test_trust_handshake_zero_or_negative_heartbeat_no_widen() -> None:
    """Pin: heartbeat <= 0 disables widening (the ``> 0`` guard).
    Without it a malformed/zero advertisement would route through the
    widen math and produce ``_read_timeout = max(t, 0) = t`` —
    behaviourally a no-op today, but the explicit guard keeps the
    diagnostic logging accurate."""
    proto = _make_protocol(timeout=5.0, trust=True)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake()

    assert proto._read_timeout == 5.0
