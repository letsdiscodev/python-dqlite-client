"""End-to-end pool concurrent-transaction + leader-flip interaction tests."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import pytest

from dqliteclient.exceptions import OperationalError
from dqliteclient.pool import ConnectionPool

if TYPE_CHECKING:
    from dqlitetestlib import TestClusterControl  # type: ignore[import-not-found]


_SQLITE_BUSY = 5
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY_S = 0.01


@pytest.mark.integration
async def test_pool_two_concurrent_writers_no_lost_rows(
    cluster_address: str,
) -> None:
    """Two concurrent write transactions both land; writers retry on BUSY.

    dqlite's authorizer denies ``PRAGMA busy_timeout``, so concurrent
    writers must handle ``SQLITE_BUSY`` (code 5) at the application layer.
    """
    pool = ConnectionPool([cluster_address], min_size=2, max_size=2)
    try:
        async with pool.acquire() as setup, setup.transaction():
            await setup.execute("DROP TABLE IF EXISTS pool_concurrent_w")
            await setup.execute(
                "CREATE TABLE pool_concurrent_w (id INTEGER PRIMARY KEY, marker TEXT)"
            )

        async def writer(marker: str) -> None:
            for attempt in range(_RETRY_MAX_ATTEMPTS):
                try:
                    async with pool.acquire() as conn, conn.transaction():
                        await conn.execute(
                            "INSERT INTO pool_concurrent_w (marker) VALUES (?)",
                            [marker],
                        )
                    return
                except OperationalError as e:
                    # Only retry SQLITE_BUSY; leader-change/transport errors propagate.
                    if e.code != _SQLITE_BUSY or attempt == _RETRY_MAX_ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(_RETRY_BASE_DELAY_S * (2**attempt))

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
    """Leader flip during an open tx: COMMIT fails, pool invalidates the slot
    on release, and the next acquire routes a fresh connection to the new leader.
    """
    starting_leader = await cluster_control.current_leader_node()

    pool = ConnectionPool(cluster_node_addresses, min_size=1, max_size=2)
    try:
        await pool.initialize()

        async with pool.acquire() as setup, setup.transaction():
            await setup.execute("DROP TABLE IF EXISTS leader_flip_t")
            await setup.execute("CREATE TABLE leader_flip_t (id INTEGER PRIMARY KEY, marker TEXT)")

        commit_error: OperationalError | None = None

        # Manual BEGIN/COMMIT (not transaction()) so we can assert COMMIT failure.
        async with pool.acquire() as conn:
            await conn.execute("BEGIN")
            await conn.execute("INSERT INTO leader_flip_t (marker) VALUES (?)", ["A"])

            flip = await cluster_control.force_leader_flip()
            assert flip.target.node_id != starting_leader.node_id
            assert flip.leader_after == flip.target.address

            try:
                await conn.execute("COMMIT")
            except OperationalError as e:
                commit_error = e

        assert commit_error is not None, (
            f"expected COMMIT to fail post-flip but it succeeded. "
            f"flip target was node {flip.target.node_id} "
            f"@ {flip.target.address}"
        )

        # Pool slot recovery: a fresh acquire must succeed against the new leader.
        async with pool.acquire() as fresh, fresh.transaction():
            await fresh.execute("INSERT INTO leader_flip_t (marker) VALUES (?)", ["C"])

        async with pool.acquire() as check:
            rows = await check.fetchall("SELECT marker FROM leader_flip_t")
            markers = {r[0] for r in rows}
            assert "C" in markers, f"post-flip INSERT did not commit; got rows {rows!r}"

        # Best-effort restore so subsequent tests start from the same leader.
        with contextlib.suppress(Exception):
            await cluster_control.transfer_leadership_to(starting_leader.node_id)
            await cluster_control.wait_for_leader_change(flip.leader_after)
    finally:
        await pool.close()
