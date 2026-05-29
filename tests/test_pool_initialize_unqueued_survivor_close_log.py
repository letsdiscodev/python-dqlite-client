"""Pin: when an unqueued-survivor close raises ``_POOL_CLEANUP_EXCEPTIONS``,
initialize logs it at DEBUG and keeps closing the rest instead of re-raising."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_unqueued_survivor_close_failure_debug_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A conn whose close() raises emits a DEBUG log without disrupting the
    rest of the cleanup loop."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=2, max_size=4)

    clean_conn = MagicMock()
    clean_conn.close = AsyncMock()
    failing_conn = MagicMock()
    failing_conn.close = AsyncMock(side_effect=OSError("simulated close failure"))

    # put_nowait raises so both conns end up on unqueued_survivors.
    create_calls = iter([clean_conn, failing_conn])

    async def stub_create() -> MagicMock:
        return next(create_calls)

    with (
        patch.object(pool, "_create_connection", new=stub_create),
        patch.object(pool._pool, "put_nowait", side_effect=asyncio.QueueFull),
        caplog.at_level("DEBUG"),
        pytest.raises(asyncio.QueueFull),
    ):
        await pool.initialize()

    clean_conn.close.assert_awaited()
    failing_conn.close.assert_awaited()
    assert any("unqueued-survivor close error" in r.message for r in caplog.records)
