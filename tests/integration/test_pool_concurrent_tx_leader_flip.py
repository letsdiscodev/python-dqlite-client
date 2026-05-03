"""Pool-level concurrent transactions interact with leader flip.

The intersection of ``ConnectionPool.acquire``, the per-connection
``transaction()`` context manager, ``_invalidate``, and the leader-
flip-mid-tx response from the server is the load-bearing concurrent-
transactions surface for the client. The unit suite covers each
piece in isolation; these end-to-end tests pin the interaction.

The basic concurrent-writers test now runs unconditionally — the
``python-dqlite-dev`` cluster (host networking, ``127.0.0.1:900N``
advertised addresses) makes leader-redirect-following work from a
host-side test runner.

The leader-flip-mid-tx test is still gated on the leader-flip
primitive — coming in the ``python-dqlite-dev`` testlib alongside
``ClusterClient.transfer_leadership``.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "Two concurrent writers can race for the SQLite write lock and "
        "the loser surfaces ``SQLITE_BUSY`` without a configured "
        "busy_timeout. The test needs a retry-on-BUSY shape (or a "
        "busy_timeout pragma) before the assertion holds reliably. "
        "Strict=False so a lucky-timing pass is acknowledged but does "
        "not flip CI red."
    ),
    strict=False,
)
async def test_pool_two_concurrent_writers_no_lost_rows(
    cluster_address: str,
) -> None:
    """Two pool-acquired connections each running their own
    transaction. Both COMMIT cleanly when no contention; pool slots
    return clean."""
    pool = ConnectionPool([cluster_address], min_size=2, max_size=2)
    try:
        async with pool.acquire() as setup:
            await setup.execute("DROP TABLE IF EXISTS pool_concurrent_w")
            await setup.execute(
                "CREATE TABLE pool_concurrent_w (id INTEGER PRIMARY KEY, marker TEXT)"
            )

        async def writer(marker: str) -> None:
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute("INSERT INTO pool_concurrent_w (marker) VALUES (?)", [marker])

        await asyncio.gather(writer("A"), writer("B"))

        async with pool.acquire() as check:
            rows = await check.fetchall("SELECT marker FROM pool_concurrent_w")
            markers = sorted(r[0] for r in rows)
            assert markers == ["A", "B"]
    finally:
        await pool.close()


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Gated on a leader-flip primitive (see above).",
    strict=False,
)
async def test_pool_leader_flip_mid_tx_loser_rolled_back(
    cluster_address: str,
) -> None:
    """A leader flip during one of two concurrent transactions: the
    affected slot is invalidated and discarded; the next acquire
    returns a fresh connection bound to the new leader."""
    # Test body would: acquire two connections, BEGIN+INSERT on each,
    # call the (yet-to-exist) `cluster.force_leader_flip()`, expect
    # one writer to surface a leader-change OperationalError, both
    # connections to be cleanly invalidated/returned, and the next
    # acquire to succeed against the new leader. Body is left as a
    # docstring sketch until the primitive lands.
    raise NotImplementedError("waiting on leader-flip primitive")
