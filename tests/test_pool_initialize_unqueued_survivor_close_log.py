"""``ConnectionPool.initialize`` walks ``unqueued_survivors`` after a
mid-init cancel and closes each. If the close itself raises
``_POOL_CLEANUP_EXCEPTIONS`` (OSError / DqliteConnectionError), the
``logger.debug`` emit at ``pool.py:623-627`` is the only forensic
trail.

Without coverage, a regression that drops the ``except`` clause and
re-raises during cleanup would supplant the user-visible original
error and the unqueued-survivor cleanup loop would abort partway
through, leaking subsequent survivors.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_unqueued_survivor_close_failure_debug_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A successful conn whose subsequent close() raises must
    emit a DEBUG log without disrupting the rest of the cleanup
    loop."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=2, max_size=4)

    # Two conns: one that closes cleanly, one that raises OSError.
    clean_conn = MagicMock()
    clean_conn.close = AsyncMock()
    failing_conn = MagicMock()
    failing_conn.close = AsyncMock(side_effect=OSError("simulated close failure"))

    # _create_connection returns the two conns in order. After they
    # succeed, force put_nowait to raise so they end up on
    # unqueued_survivors.
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

    # Both close() calls must have been attempted; the OSError on
    # the second is logged but does not stop the loop.
    clean_conn.close.assert_awaited()
    failing_conn.close.assert_awaited()
    assert any("unqueued-survivor close error" in r.message for r in caplog.records)
