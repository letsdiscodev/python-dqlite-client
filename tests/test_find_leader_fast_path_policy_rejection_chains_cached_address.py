"""Pin: ``ClusterClient.find_leader``'s cached fast-path policy-
rejection arm chains the raised ``ClusterPolicyError`` via ``__cause__``
back to the cached address that triggered the redirect, so a forensic
walker can correlate "which cached node redirected us to the rejected
target".

Pre-fix the fast-path arm did ``raise`` (no chain), so when the cached
node's redirect landed on a policy-rejected target the propagated
exception carried no breadcrumb back to the cached source. The parallel-
sweep arm already accumulates per-node history via the sweep-side
chaining at ``cluster.py:1349-1356`` — the fast-path-only path was
the asymmetric gap.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore

pytestmark = pytest.mark.asyncio


async def test_find_leader_fast_path_policy_rejection_chains_cached_address() -> None:
    store = MemoryNodeStore(["10.0.0.1:9001", "10.0.0.2:9001"])
    # redirect_policy rejects 10.0.0.99 explicitly. The cached node
    # at 10.0.0.1 will redirect to 10.0.0.99 — _check_redirect raises.
    client = ClusterClient(
        store,
        redirect_policy=lambda addr: addr != "10.0.0.99:9001",
    )
    # Seed the cache so find_leader takes the fast path.
    client._set_last_known_leader("10.0.0.1:9001")
    # Mock _query_leader on the fast-path arm: return the rejected
    # redirect target so _check_redirect raises ClusterPolicyError.
    client._query_leader = AsyncMock(return_value="10.0.0.99:9001")

    with pytest.raises(ClusterPolicyError) as excinfo:
        await client.find_leader()

    # Chain pin: the raised ClusterPolicyError carries a __cause__
    # naming the cached source address so an operator chasing
    # "which cached node redirected us?" finds it in the exception
    # chain rather than having to read logs separately.
    assert excinfo.value.__cause__ is not None, (
        "fast-path policy-rejection raise must chain via __cause__ "
        "so the cached-address breadcrumb is preserved"
    )
    cause = excinfo.value.__cause__
    assert isinstance(cause, ClusterError), (
        f"expected ClusterError cause naming the cached source; got {type(cause).__name__}"
    )
    assert "10.0.0.1:9001" in str(cause), f"cause text must name the cached address; got {cause!s}"
