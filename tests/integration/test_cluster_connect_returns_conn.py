"""Pin the success-path return of ``ClusterClient.connect()`` against the real cluster,
patching ``find_leader`` to return the host-reachable address (leader-discovery otherwise
returns container-internal addresses unreachable from the docker-host runner)."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection


@pytest.mark.integration
class TestClusterConnectReturnsConn:
    async def test_connect_returns_initialized_dqlite_connection(
        self, monkeypatch: pytest.MonkeyPatch, cluster_address: str
    ) -> None:
        async def _patched_find_leader(
            self: ClusterClient,
            *,
            trust_server_heartbeat: bool = False,
            policy: object = None,
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
