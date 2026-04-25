"""Pool-level concurrent transactions interact with leader flip.

The intersection of ``ConnectionPool.acquire``, the per-connection
``transaction()`` context manager, ``_invalidate``, and the leader-
flip-mid-tx response from the server is the load-bearing concurrent-
transactions surface for the client. The unit suite covers each
piece in isolation; an end-to-end test pinning the interaction is
missing.

Two gating dependencies prevent these tests from running today
against the existing ``dqlite-test-cluster`` fixture:

1. The pool's leader-find resolves the redirect address advertised
   by the cluster nodes; in the docker-compose setup each container
   advertises ``0.0.0.0:9001`` (its container-internal binding) which
   is unreachable from the docker-host test runner. Direct
   ``connect()`` works because the docker port-mapping serves the
   exposed external ports; the pool's redirect-following does not.

2. There is no leader-flip primitive in the test fixtures —
   ``cluster.transfer()`` over the wire-protocol TRANSFER request
   is plumbed in dqlite but not exposed as a pytest fixture.

Pin the test shape now so the moment the cluster fixture provides
both reachable redirect addresses and a leader-flip primitive, the
xfail marker can be removed.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "Gated on cluster-fixture work: pool leader-find chases "
        "container-internal addresses (0.0.0.0:9001) that are not "
        "reachable from the docker-host test runner; no leader-flip "
        "primitive in the test fixtures. Pinned for unblocking once "
        "both fixtures land."
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
