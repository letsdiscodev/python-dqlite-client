"""A server-redirected leader address that fails _canonicalize_host surfaces as
ClusterPolicyError, not bare ValueError: connect()'s narrow except would let a
ValueError leak past, bypassing PEP 249 wrapping and is_disconnect classification.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_server_redirect_to_invalid_address_raises_cluster_policy_error() -> None:
    """An address with control bytes fails _canonicalize_host; the wrap holds."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        return "host\nnewline:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_server_redirect_to_non_ascii_address_raises_cluster_policy_error() -> None:
    """Non-ASCII host (IDN) is rejected; the wrap holds."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        return "résumé.example.com:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_server_redirect_to_oversized_hostname_raises_cluster_policy_error() -> None:
    """Hostname > 253 chars exceeds the DNS limit; the wrap holds."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    oversize = "a" * 260 + ".example.com:9001"

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        return oversize

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()
