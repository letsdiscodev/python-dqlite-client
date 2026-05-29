"""Pin: ``dqliteclient.connect()`` closes the partial
``DqliteConnection`` when ``connect()`` raises, so loop-bound asyncio
primitives are not orphaned until GC.
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
        patch.object(DqliteConnection, "connect", new=AsyncMock(side_effect=OSError("boom"))),
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
    logged (with exc_info) so the original connect error propagates."""
    import logging

    with (
        patch.object(DqliteConnection, "connect", new=AsyncMock(side_effect=OSError("boom"))),
        patch.object(
            DqliteConnection,
            "close",
            new=AsyncMock(side_effect=RuntimeError("close-also-broken")),
        ),
        caplog.at_level(logging.DEBUG, logger="dqliteclient"),
        pytest.raises(OSError, match="boom"),
    ):
        await dqliteclient.connect("localhost:9001", database="test", timeout=5.0)

    debug_rec = next(
        (r for r in caplog.records if "cleanup-close after failed connect" in r.getMessage()),
        None,
    )
    assert debug_rec is not None, (
        "expected a DEBUG record from the connect cleanup-close suppression arm"
    )
    assert debug_rec.levelno == logging.DEBUG, (
        "cleanup-close suppression must log at DEBUG level — a refactor that "
        "bumps to INFO/WARNING surfaces the secondary failure too loudly"
    )
    assert debug_rec.exc_info is not None, (
        "DEBUG record must carry exc_info so operators can see why cleanup failed"
    )
    assert isinstance(debug_rec.exc_info[1], RuntimeError)
    assert "close-also-broken" in str(debug_rec.exc_info[1])


@pytest.mark.asyncio
async def test_public_connect_baseexception_propagates_after_cleanup() -> None:
    """A ``BaseException`` during connect() still drives cleanup before
    propagating, so loop-bound state is not leaked under outer cancel."""
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
