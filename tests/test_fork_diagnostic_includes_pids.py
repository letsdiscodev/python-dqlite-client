"""Pin: fork-after-init diagnostics on ``DqliteConnection``,
``ConnectionPool``, and ``ClusterClient`` include both the creator
pid and the current observed pid so an operator can correlate the
failure to the master / forkserver pid.

Symmetric with the cross-thread sibling diagnostic at the dbapi
layer which already includes both ids; without these pids the
fork-rejection messages were a constant string with no operator-
correlation surface.
"""

from __future__ import annotations

import re
from unittest import mock

import pytest

from dqliteclient import (
    ClusterClient,
    ConnectionPool,
    DqliteConnection,
    InterfaceError,
    MemoryNodeStore,
)


def test_dqlite_connection_fork_diagnostic_includes_creator_and_current_pid() -> None:
    conn = DqliteConnection("localhost:9001")
    fake_child_pid = conn._creator_pid + 1
    with (
        mock.patch("dqliteclient.connection.os.getpid", return_value=fake_child_pid),
        mock.patch("dqliteclient.connection.get_current_pid", return_value=fake_child_pid),
        pytest.raises(InterfaceError) as exc,
    ):
        conn._check_in_use()
    msg = str(exc.value)
    assert "used after fork" in msg
    assert re.search(rf"pid {conn._creator_pid}\b", msg), msg
    assert re.search(rf"pid {fake_child_pid}\b", msg), msg


def test_connection_pool_initialize_fork_diagnostic_includes_pids() -> None:
    pool = ConnectionPool(addresses=["localhost:9001"])
    fake_child_pid = pool._creator_pid + 1
    with (
        mock.patch("dqliteclient.connection.os.getpid", return_value=fake_child_pid),
        mock.patch("dqliteclient.connection.get_current_pid", return_value=fake_child_pid),
    ):
        import asyncio

        async def _run() -> None:
            with pytest.raises(InterfaceError) as exc:
                await pool.initialize()
            msg = str(exc.value)
            assert "used after fork" in msg
            assert re.search(rf"pid {pool._creator_pid}\b", msg), msg
            assert re.search(rf"pid {fake_child_pid}\b", msg), msg

        asyncio.run(_run())


def test_cluster_client_find_leader_fork_diagnostic_includes_pids() -> None:
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]))
    fake_child_pid = cluster._creator_pid + 1
    with (
        mock.patch("dqliteclient.connection.os.getpid", return_value=fake_child_pid),
        mock.patch("dqliteclient.connection.get_current_pid", return_value=fake_child_pid),
    ):
        import asyncio

        async def _run() -> None:
            with pytest.raises(InterfaceError) as exc:
                await cluster.find_leader()
            msg = str(exc.value)
            assert "used after fork" in msg
            assert re.search(rf"pid {cluster._creator_pid}\b", msg), msg
            assert re.search(rf"pid {fake_child_pid}\b", msg), msg

        asyncio.run(_run())
