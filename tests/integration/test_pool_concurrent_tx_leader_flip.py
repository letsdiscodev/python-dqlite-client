"""Pool-level concurrent transactions interact with leader flip.

The intersection of ``ConnectionPool.acquire``, the per-connection
``transaction()`` context manager, ``_invalidate``, and the leader-
flip-mid-tx response from the server is the load-bearing concurrent-
transactions surface for the client. The unit suite covers each
piece in isolation; these end-to-end tests pin the interaction.

Both tests run against the python-dqlite-dev cluster (host
networking + ``127.0.0.1:900N`` advertised addresses); the
leader-flip test additionally uses the
``cluster_control`` fixture from ``dqlitetestlib``, bootstrapped
by the top-level ``tests/conftest.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import pytest

from dqliteclient.exceptions import OperationalError
from dqliteclient.pool import ConnectionPool

if TYPE_CHECKING:
    from dqlitetestlib import TestClusterControl  # type: ignore[import-not-found]


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
async def test_pool_leader_flip_mid_tx_loser_rolled_back(
    cluster_node_addresses: list[str],
    cluster_control: TestClusterControl,
) -> None:
    """A leader flip during an open transaction: the pending COMMIT
    on the old leader fails with ``OperationalError`` (Raft has
    demoted the server), the pool invalidates the broken slot on
    release, and the next acquire returns a fresh connection bound
    to the new leader.

    Despite the file's "two concurrent transactions" framing, the
    load-bearing contract here is the per-slot invalidation +
    recovery sequence — and one held transaction demonstrates it
    cleanly. Two SQLite writers contending for the BEGIN/INSERT
    write-lock would surface ``SQLITE_BUSY`` from a parallel
    operation rather than the leader-change error we want to
    observe (see the ``test_pool_two_concurrent_writers_no_lost_rows``
    xfail on the same lock-contention issue).

    Test shape:

    1. Snapshot the starting leader so we can restore on teardown.
    2. Acquire one pool connection, ``BEGIN`` + ``INSERT``.
    3. Force a leader transfer via ``cluster_control``. The new
       leader is a different voter; the old leader has stepped down.
    4. ``COMMIT`` on the held connection fails with
       ``OperationalError`` (the old leader rejects further writes
       — typically ``SQLITE_IOERR_NOT_LEADER``).
    5. A fresh ``pool.acquire()`` succeeds: the pool re-runs
       ``find_leader``, gets the new leader's address, and routes a
       new connection there. A trivial ``INSERT`` + ``SELECT``
       confirms the freshly-acquired connection can both write and
       observe its own write.
    6. Restore the original leader on the way out so subsequent
       tests see a deterministic starting state.
    """
    starting_leader = await cluster_control.current_leader_node()

    pool = ConnectionPool(cluster_node_addresses, min_size=1, max_size=2)
    try:
        await pool.initialize()

        async with pool.acquire() as setup, setup.transaction():
            await setup.execute("DROP TABLE IF EXISTS leader_flip_t")
            await setup.execute("CREATE TABLE leader_flip_t (id INTEGER PRIMARY KEY, marker TEXT)")

        commit_error: OperationalError | None = None

        # Hold the connection + open transaction across the leader
        # flip. Manual BEGIN/COMMIT (rather than ``transaction()``)
        # so we can assert on the COMMIT failure shape.
        async with pool.acquire() as conn:
            await conn.execute("BEGIN")
            await conn.execute("INSERT INTO leader_flip_t (marker) VALUES (?)", ["A"])

            # Force the current leader to step down and a different
            # voter to take over. The pending transaction is bound
            # to the old leader's TCP slot — it cannot follow the
            # transfer.
            flip = await cluster_control.force_leader_flip()
            assert flip.target.node_id != starting_leader.node_id
            assert flip.leader_after == flip.target.address

            try:
                await conn.execute("COMMIT")
            except OperationalError as e:
                commit_error = e

        # The held tx targeted the (now stepped-down) leader; the
        # COMMIT must fail.
        assert commit_error is not None, (
            f"expected COMMIT to fail post-flip but it succeeded. "
            f"flip target was node {flip.target.node_id} "
            f"@ {flip.target.address}"
        )

        # Pool slot recovery: the broken connection was released on
        # the ``async with`` exit; a fresh acquire must succeed
        # against the new leader.
        async with pool.acquire() as fresh, fresh.transaction():
            await fresh.execute("INSERT INTO leader_flip_t (marker) VALUES (?)", ["C"])

        async with pool.acquire() as check:
            rows = await check.fetchall("SELECT marker FROM leader_flip_t")
            markers = {r[0] for r in rows}
            assert "C" in markers, f"post-flip INSERT did not commit; got rows {rows!r}"

        # Restore: transfer leadership back so subsequent tests start
        # from the same leader. Best-effort — failure here does not
        # invalidate the load-bearing assertions above.
        with contextlib.suppress(Exception):
            await cluster_control.transfer_leadership_to(starting_leader.node_id)
            await cluster_control.wait_for_leader_change(flip.leader_after)
    finally:
        await pool.close()
