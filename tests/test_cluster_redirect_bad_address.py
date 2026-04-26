"""Pin: server-redirected leader address that fails ``_canonicalize_host``
validation surfaces as ``ClusterPolicyError`` (not bare ``ValueError``).

The leader address comes from the wire (LeaderResponse). A hostile or
buggy server can redirect to an address that doesn't pass our
hostname / IP / IDN / character-class validation. ``DqliteConnection``
raises bare ``ValueError`` for those; the cluster.connect() retry
loop's narrow ``except (OSError, DqliteConnectionError, ClusterError)``
would let the ValueError leak past, bypassing PEP 249 exception
wrapping at higher layers.

Wrap as ``ClusterPolicyError`` so:
- The SA dialect's ``is_disconnect`` correctly classifies it as
  non-retryable (not a transient transport failure).
- The dbapi layer's PEP 249 wrap sees a known error class.
- Operators reading logs see "invalid leader address" diagnostics
  instead of an unrelated ValueError.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_server_redirect_to_invalid_address_raises_cluster_policy_error() -> None:
    """A leader address with embedded NUL / control bytes fails
    ``_canonicalize_host`` with bare ValueError. Pin the wrap."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False) -> str:
        # Hostile address with embedded control character — fails
        # _canonicalize_host's "no whitespace/CR/LF" check (since it
        # also fails the hostname-label regex).
        return "host\nnewline:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_server_redirect_to_non_ascii_address_raises_cluster_policy_error() -> None:
    """Non-ASCII host (IDN) is rejected; pin the wrap."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False) -> str:
        return "résumé.example.com:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_server_redirect_to_oversized_hostname_raises_cluster_policy_error() -> None:
    """Hostname > 253 chars exceeds DNS limit; pin the wrap."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    oversize = "a" * 260 + ".example.com:9001"

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False) -> str:
        return oversize

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()
