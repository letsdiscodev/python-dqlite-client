"""Pin: ``ConnectionPool.close()`` clears the at-capacity wakeup
event ``self._closed_event`` after the drain completes.

The pool's close path nulls every other once-used loop-bound
field after use (``_async_conn``, ``_protocol``, ``_connect_lock``,
``_op_lock``, ``_loop_ref``); ``_closed_event`` is the lone
holdout. A signalled-and-now-useless ``asyncio.Event`` bound to
the pool's loop is held until the pool object itself is GC'd,
keeping the loop reference alive past the user's intended
lifetime when the pool object outlives ``close()`` (SA engine
diagnostic accessors, module-level state, etc.).

``_close_done`` is deliberately NOT cleared — see the issue's
"Subtle" paragraph: clearing in the second-caller branch is
TOCTOU; clearing only in the first caller's finally is what's
needed, but the second-caller arm awaits the same event so we
keep its lifetime tied to the pool's own.
"""

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
    # Force the wakeup event to exist (acquire normally lazy-creates
    # it via ``_get_closed_event`` when a parker is going to sleep).
    pool._get_closed_event()
    assert pool._closed_event is not None

    await pool.close()

    # Mirror the discipline applied to every other once-used
    # loop-bound field on the close path: drop the now-useless
    # event reference so it can be GC'd alongside the pool's
    # other loop primitives. ``_close_done`` stays — the
    # second-caller arm awaits it.
    assert pool._closed_event is None, (
        "ConnectionPool.close() must clear _closed_event after "
        "set() — every other loop-bound field on the close path "
        "is nulled; this is the lone exception, leaving a "
        "signalled Event pinned to the pool's loop."
    )
