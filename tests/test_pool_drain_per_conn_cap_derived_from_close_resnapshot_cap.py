"""Pin: the per-connection drain cap is derived from
``_CLOSE_RESNAPSHOT_CAP + 1`` (the inner ``_close_impl`` worst-case is
``(cap + 1) × close_timeout``) rather than the old ``close_timeout + 0.5``
literal, keeping the two drain paths in lockstep with the resnapshot cap."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import _CLOSE_RESNAPSHOT_CAP
from dqliteclient.pool import _DRAIN_PER_CONN_CAP_MULTIPLIER, ConnectionPool


def test_drain_per_conn_cap_multiplier_derives_from_close_resnapshot_cap() -> None:
    assert _DRAIN_PER_CONN_CAP_MULTIPLIER == _CLOSE_RESNAPSHOT_CAP + 1


def test_drain_per_conn_cap_multiplier_is_named_not_magic() -> None:
    """The drain cap must not reappear as a bare ``+ 0.5`` literal."""
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
    """A single hung ``conn.close()`` must not block the rest of the drain:
    the per-iteration cap fires and the loop moves on."""
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

    # Per-iter cap = 0.05 * (_CLOSE_RESNAPSHOT_CAP + 1) = 0.2s, so the drain
    # finishes within 2s even with conn_b hung.
    async with asyncio.timeout(2.0):
        await pool._drain_idle()

    conn_a.close.assert_awaited()
    conn_c.close.assert_awaited()
    hang_evt.set()  # release the hung close so it doesn't leak
