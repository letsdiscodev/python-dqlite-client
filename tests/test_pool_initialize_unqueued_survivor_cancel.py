"""Pin the CancelledError arm of initialize's unqueued-survivor cleanup: a
cancel mid-close must keep closing the remaining survivors before resuming."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_unqueued_survivor_cancel_continues_closing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CancelledError in one survivor's close emits the DEBUG breadcrumb and
    still closes the remaining survivors before the cancel resumes."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=2, max_size=4)

    # First conn's close() raises CancelledError; the second must still close.
    cancelled_conn = MagicMock()
    cancelled_conn.close = AsyncMock(side_effect=asyncio.CancelledError())
    clean_conn = MagicMock()
    clean_conn.close = AsyncMock()

    create_calls = iter([cancelled_conn, clean_conn])

    async def stub_create() -> MagicMock:
        return next(create_calls)

    with (
        patch.object(pool, "_create_connection", new=stub_create),
        patch.object(pool._pool, "put_nowait", side_effect=asyncio.QueueFull),
        caplog.at_level("DEBUG"),
        pytest.raises(asyncio.QueueFull),
    ):
        await pool.initialize()

    cancelled_conn.close.assert_awaited()
    clean_conn.close.assert_awaited()

    assert any("cancel during unqueued-survivor close" in r.getMessage() for r in caplog.records), (
        "expected the cancel-arm DEBUG breadcrumb at pool.py:889 "
        "'cancel during unqueued-survivor close'"
    )
