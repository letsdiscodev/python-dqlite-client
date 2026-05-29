"""``_query_leader`` propagates pre-handshake transport errors so the aggregate
ClusterError distinguishes "node unreachable" from "node up, no leader elected"."""

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_query_leader_propagates_connection_refused() -> None:
    cluster = ClusterClient(MemoryNodeStore(["unreachable:9001"]), timeout=2.0)

    async def refused_open(*_args: object, **_kwargs: object):
        raise ConnectionRefusedError("nothing listening")

    with (
        patch(
            "dqliteclient._dial.open_connection_with_keepalive",
            side_effect=refused_open,
        ),
        pytest.raises(ConnectionRefusedError),
    ):
        await cluster._query_leader("unreachable:9001")


@pytest.mark.asyncio
async def test_query_leader_propagates_timeout() -> None:
    cluster = ClusterClient(
        MemoryNodeStore(["slow:9001"]),
        timeout=2.0,
        dial_timeout=0.05,
    )

    async def slow_open(*_args: object, **_kwargs: object):
        await asyncio.sleep(1.0)

    with (
        patch(
            "dqliteclient._dial.open_connection_with_keepalive",
            side_effect=slow_open,
        ),
        pytest.raises(TimeoutError),
    ):
        await cluster._query_leader("slow:9001")


@pytest.mark.asyncio
async def test_find_leader_aggregate_attributes_dial_error() -> None:
    """A refused node appears in the aggregate ClusterError with its specific class."""
    cluster = ClusterClient(
        MemoryNodeStore(["unreachable:9001"]),
        timeout=2.0,
        dial_timeout=0.5,
        attempt_timeout=1.0,
    )

    async def refused_open(*_args: object, **_kwargs: object):
        raise ConnectionRefusedError("nothing listening")

    with (
        patch(
            "dqliteclient._dial.open_connection_with_keepalive",
            side_effect=refused_open,
        ),
        pytest.raises(ClusterError) as excinfo,
    ):
        await cluster.find_leader()

    msg = str(excinfo.value)
    assert "unreachable:9001" in msg
    assert "ConnectionRefusedError" in msg or "refused" in msg.lower() or "nothing listening" in msg
