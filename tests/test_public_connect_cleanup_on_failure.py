"""Pin: ``dqliteclient.connect()`` cleans up the partial
``DqliteConnection`` when ``connect()`` raises, mirroring the
``dqlitedbapi.aconnect`` discipline.

Pre-fix the eager ``await conn.connect()`` had no try/except, so a
failure during dial / handshake left the partially-bound asyncio
primitives (loop-bound locks, transport handles, reader task)
referenced only by the now-orphaned ``conn`` until GC. The dbapi
sibling ``aconnect`` and the SA ``DqliteDialect_aio.connect`` both
already wrap the partial-state cleanup; this test pins the parity
on the client public API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import dqliteclient
from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_public_connect_failure_calls_close_on_partial_conn() -> None:
    close_calls: list[None] = []

    real_close = DqliteConnection.close

    async def _tracking_close(self: DqliteConnection) -> None:
        close_calls.append(None)
        await real_close(self)

    with (
        patch.object(
            DqliteConnection, "connect", new=AsyncMock(side_effect=OSError("boom"))
        ),
        patch.object(DqliteConnection, "close", new=_tracking_close),
        pytest.raises(OSError, match="boom"),
    ):
        await dqliteclient.connect("localhost:9001", database="test", timeout=5.0)

    assert close_calls, (
        "dqliteclient.connect() must call close() on the partially-constructed "
        "DqliteConnection when connect() raises"
    )


@pytest.mark.asyncio
async def test_public_connect_close_failure_during_cleanup_logged_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A close-time exception during cleanup is suppressed and DEBUG-
    logged so the original connect error propagates unchanged."""
    import logging

    with (
        patch.object(
            DqliteConnection, "connect", new=AsyncMock(side_effect=OSError("boom"))
        ),
        patch.object(
            DqliteConnection,
            "close",
            new=AsyncMock(side_effect=RuntimeError("close-also-broken")),
        ),
        caplog.at_level(logging.DEBUG, logger="dqliteclient"),
        pytest.raises(OSError, match="boom"),
    ):
        await dqliteclient.connect("localhost:9001", database="test", timeout=5.0)


@pytest.mark.asyncio
async def test_public_connect_baseexception_propagates_after_cleanup() -> None:
    """A ``BaseException`` (e.g. CancelledError-shaped) during connect()
    must still drive cleanup before propagating, so loop-bound state
    isn't leaked under outer ``asyncio.timeout`` cancellation."""
    close_calls: list[None] = []

    real_close = DqliteConnection.close

    async def _tracking_close(self: DqliteConnection) -> None:
        close_calls.append(None)
        await real_close(self)

    with (
        patch.object(
            DqliteConnection,
            "connect",
            new=AsyncMock(side_effect=KeyboardInterrupt("synthetic")),
        ),
        patch.object(DqliteConnection, "close", new=_tracking_close),
        pytest.raises(KeyboardInterrupt, match="synthetic"),
    ):
        await dqliteclient.connect("localhost:9001", database="test", timeout=5.0)

    assert close_calls, (
        "dqliteclient.connect() must call close() on the partial conn "
        "even on BaseException propagation"
    )
