"""A ValueError from a client-side knob must not be misattributed as "Server
redirected to invalid leader address"; only address-shape failures earn that
prefix, or operator triage points the wrong way."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_close_timeout_zero_raises_value_error_not_cluster_policy_error() -> None:
    """close_timeout=0 surfaces as a knob ValueError, not a server-redirect error."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        # Valid leader so the address pre-check passes and we reach the knob validator.
        return "localhost:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ValueError) as excinfo,
    ):
        await client.connect(close_timeout=0)

    assert "close_timeout" in str(excinfo.value)
    assert "Server redirected" not in str(excinfo.value)
