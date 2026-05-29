"""The cached fast-path policy-rejection arm chains the raised
``ClusterPolicyError`` via ``__cause__`` back to the cached address that
redirected, so a forensic walker can correlate the redirect source."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore

pytestmark = pytest.mark.asyncio


async def test_find_leader_fast_path_policy_rejection_chains_cached_address() -> None:
    store = MemoryNodeStore(["10.0.0.1:9001", "10.0.0.2:9001"])
    # The cached node at 10.0.0.1 redirects to the rejected 10.0.0.99.
    client = ClusterClient(
        store,
        redirect_policy=lambda addr: addr != "10.0.0.99:9001",
    )
    # Seed the cache so find_leader takes the fast path.
    client._set_last_known_leader("10.0.0.1:9001")
    client._query_leader = AsyncMock(return_value="10.0.0.99:9001")

    with pytest.raises(ClusterPolicyError) as excinfo:
        await client.find_leader()

    assert excinfo.value.__cause__ is not None, (
        "fast-path policy-rejection raise must chain via __cause__ "
        "so the cached-address breadcrumb is preserved"
    )
    cause = excinfo.value.__cause__
    assert isinstance(cause, ClusterError), (
        f"expected ClusterError cause naming the cached source; got {type(cause).__name__}"
    )
    assert "10.0.0.1:9001" in str(cause), f"cause text must name the cached address; got {cause!s}"
