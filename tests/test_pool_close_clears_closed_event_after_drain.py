"""close() clears _closed_event after the drain, like every other once-used
loop-bound field, so a signalled Event does not pin the pool's loop past close().
_close_done is deliberately NOT cleared: the second-caller arm still awaits it."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_close_clears_closed_event_after_drain() -> None:
    cluster = MagicMock(spec=ClusterClient)
    pool = ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=2,
        timeout=1.0,
        cluster=cluster,
    )
    # Force the wakeup event to exist (acquire lazy-creates it for parkers).
    pool._get_closed_event()
    assert pool._closed_event is not None

    await pool.close()

    assert pool._closed_event is None, (
        "ConnectionPool.close() must clear _closed_event after "
        "set() — every other loop-bound field on the close path "
        "is nulled; this is the lone exception, leaving a "
        "signalled Event pinned to the pool's loop."
    )
