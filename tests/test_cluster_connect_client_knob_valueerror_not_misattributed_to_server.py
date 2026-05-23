"""Pin: a ``ValueError`` raised by ``DqliteConnection.__init__`` for a
client-side knob (close_timeout, max_total_rows, max_message_size, ...)
must NOT be misattributed as "Server redirected to invalid leader
address". Only address-shape failures earn that prefix.

Before this fix the wrap caught every ``ValueError`` at the
construction site and labelled it as a server redirect, sending
operator triage in the wrong direction.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_close_timeout_zero_raises_value_error_not_cluster_policy_error() -> None:
    """Passing close_timeout=0 to ClusterClient.connect must surface
    as a ``ValueError`` from the client-knob validator — NOT wrapped
    as ClusterPolicyError("Server redirected to invalid leader
    address") which would falsely implicate the server."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        # Return a valid leader so the address pre-check passes and we
        # reach DqliteConnection.__init__ where close_timeout=0 trips
        # the knob validator.
        return "localhost:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ValueError) as excinfo,
    ):
        await client.connect(close_timeout=0)

    # The error message must reference the knob, not the server.
    assert "close_timeout" in str(excinfo.value)
    assert "Server redirected" not in str(excinfo.value)
