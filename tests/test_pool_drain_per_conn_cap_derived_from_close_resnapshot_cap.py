"""Pin: per-connection drain cap on the after-cancel sweep AND the
main ``_drain_idle`` loop is derived from
``connection._CLOSE_RESNAPSHOT_CAP + 1`` rather than the previous
``close_timeout + 0.5`` magic literal.

The inner ``DqliteConnection._close_impl`` worst-case wall-clock is
``(_CLOSE_RESNAPSHOT_CAP + 1) × close_timeout``: each re-snapshot
iteration (up to the cap) awaits a bounded ``_pending_drain`` task
plus the final ``wait_closed``, each bounded by ``close_timeout``.
The previous ``+ 0.5`` literal was too small for any
``close_timeout > 0.17`` (truncating progress on the otherwise-
correct graceful drain) and 50× the budget at the ``close_timeout
= 0.01`` floor (defeating tight-shutdown SLOs).

Deriving from the module-level constant keeps the two drain paths
in lockstep and surfaces a future bump to the resnapshot cap
automatically.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import _CLOSE_RESNAPSHOT_CAP
from dqliteclient.pool import _DRAIN_PER_CONN_CAP_MULTIPLIER, ConnectionPool


def test_drain_per_conn_cap_multiplier_derives_from_close_resnapshot_cap() -> None:
    """The pool's per-iteration drain cap is derived from the
    connection module's re-snapshot cap. A future bump to one must
    update the other."""
    assert _DRAIN_PER_CONN_CAP_MULTIPLIER == _CLOSE_RESNAPSHOT_CAP + 1


def test_drain_per_conn_cap_multiplier_is_named_not_magic() -> None:
    """Source-inspection sanity: the pool's drain cap must not be a
    bare ``+ 0.5`` literal anywhere. The previous magic-number
    sized the cap at the wrong scale for both the ``close_timeout
    = 0.01`` floor and ``close_timeout >= 0.5`` WAN deployments."""
    import inspect

    from dqliteclient import pool as pool_mod

    src = inspect.getsource(pool_mod)
    assert "self._close_timeout + 0.5" not in src, (
        "Magic ``+ 0.5`` literal must not re-appear — use "
        "``_DRAIN_PER_CONN_CAP_MULTIPLIER`` derived from "
        "``_CLOSE_RESNAPSHOT_CAP``."
    )
    assert "_DRAIN_PER_CONN_CAP_MULTIPLIER" in src


@pytest.mark.asyncio
async def test_drain_idle_isolates_per_connection_close_timeout_cap() -> None:
    """A single hung ``conn.close()`` must not block the rest of the
    drain. The new per-iteration ``wait_for`` envelope on
    ``_drain_idle`` mirrors the after-cancel sweep's discipline:
    the cap fires, the WARN logs, and the loop moves on."""
    import asyncio

    pool = ConnectionPool(["127.0.0.1:9001"], max_size=3, close_timeout=0.05)

    conn_a = MagicMock()
    conn_a._pool_released = True
    conn_a._address = "h:9001"
    conn_a.close = AsyncMock(return_value=None)

    hang_evt = asyncio.Event()

    async def _hung_close() -> None:
        await hang_evt.wait()

    conn_b = MagicMock()
    conn_b._pool_released = True
    conn_b._address = "h:9002"
    conn_b.close = AsyncMock(side_effect=_hung_close)

    conn_c = MagicMock()
    conn_c._pool_released = True
    conn_c._address = "h:9003"
    conn_c.close = AsyncMock(return_value=None)

    pool._pool.put_nowait(conn_a)
    pool._pool.put_nowait(conn_b)
    pool._pool.put_nowait(conn_c)
    pool._size = 3

    # The per-iter cap is close_timeout * multiplier =
    # 0.05 * (_CLOSE_RESNAPSHOT_CAP + 1) = 0.2s — drain finishes
    # well within 2s even with conn_b hung indefinitely.
    async with asyncio.timeout(2.0):
        await pool._drain_idle()

    # All three closes started; conn_a and conn_c completed, conn_b
    # was abandoned by the cap.
    conn_a.close.assert_awaited()
    conn_c.close.assert_awaited()
    hang_evt.set()  # release the hung close so it doesn't leak
