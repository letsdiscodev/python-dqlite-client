"""Pin the success-path return of ``ClusterClient.connect()``
(``cluster.py:430``).

The integration tests for ``connect()`` go through the
module-level ``dqliteclient.connect()`` helper, which builds a
``DqliteConnection`` directly and never visits
``ClusterClient.connect()``. ``create_pool`` does visit it, but the
existing integration ``create_pool`` test is gated on cluster-fixture
work (the leader-discovery returns container-internal addresses that
are not reachable from the docker-host runner).

This test reaches the success-path return at cluster.py:430 by
patching ``find_leader`` to bypass the broken leader-discovery and
return the host-port address that IS reachable. The
``DqliteConnection.connect()`` call then runs against the real
cluster.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection


@pytest.mark.integration
class TestClusterConnectReturnsConn:
    async def test_connect_returns_initialized_dqlite_connection(
        self, monkeypatch: pytest.MonkeyPatch, cluster_address: str
    ) -> None:
        """Drives cluster.py:430 — the ``return conn`` after
        ``find_leader`` succeeds and ``DqliteConnection.connect()``
        succeeds. Patch ``find_leader`` to return the
        host-reachable address (sidesteps the
        container-internal-address bug)."""

        async def _patched_find_leader(
            self: ClusterClient, *, trust_server_heartbeat: bool = False
        ) -> str:
            return cluster_address

        monkeypatch.setattr(ClusterClient, "find_leader", _patched_find_leader)

        cluster = ClusterClient([cluster_address], timeout=2.0)  # type: ignore[arg-type]
        conn = await cluster.connect()
        try:
            assert isinstance(conn, DqliteConnection)
            assert conn.is_connected is True
        finally:
            await conn.close()
