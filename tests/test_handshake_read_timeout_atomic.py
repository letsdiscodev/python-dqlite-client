"""Pin: ``handshake`` leaves ``_read_timeout`` unchanged unless
``trust_server_heartbeat=True``, in which case it widens up to
``_HEARTBEAT_READ_TIMEOUT_CAP_SECONDS`` and never narrows. The default must not
widen, else a hostile server could stretch the operator's read SLO to 300 s."""

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
    """Default ``trust_server_heartbeat=False`` does not widen the read deadline."""
    proto = _make_protocol(timeout=5.0, trust=False)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=300_000))

    await proto.handshake()

    assert proto._read_timeout == 5.0
    # Heartbeat is still recorded for diagnostics.
    assert proto._heartbeat_timeout == 300_000


@pytest.mark.asyncio
async def test_trust_handshake_widens_up_to_cap() -> None:
    """``trust=True`` with a heartbeat above the cap widens to the cap, not beyond."""
    proto = _make_protocol(timeout=5.0, trust=True)
    proto._send = AsyncMock()
    # Advertise 600 s; cap is 300.
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=600_000))

    await proto.handshake()

    assert proto._read_timeout == _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS


@pytest.mark.asyncio
async def test_trust_handshake_does_not_narrow_read_timeout() -> None:
    """A small heartbeat does not narrow the operator's larger configured timeout."""
    proto = _make_protocol(timeout=30.0, trust=True)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=1_000))

    await proto.handshake()

    assert proto._read_timeout == 30.0


@pytest.mark.asyncio
async def test_trust_handshake_zero_or_negative_heartbeat_no_widen() -> None:
    """Heartbeat <= 0 disables widening (the ``> 0`` guard keeps diagnostics accurate)."""
    proto = _make_protocol(timeout=5.0, trust=True)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake()

    assert proto._read_timeout == 5.0
