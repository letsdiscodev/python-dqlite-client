"""Pin: the ``max_attempts`` validation error wording is identical
in ``ConnectionPool.__init__`` and ``ClusterClient.connect``.

Operators triaging via grep should not have to remember which layer
validated first; the same constraint must produce the same error
message.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_pool_max_attempts_validation_message(bad_value: int) -> None:
    with pytest.raises(ValueError, match="must be at least 1") as exc_info:
        ConnectionPool(["a:9001"], max_attempts=bad_value)
    assert f"got {bad_value}" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [0, -1, -100])
async def test_cluster_max_attempts_validation_message(bad_value: int) -> None:
    cluster = ClusterClient(MemoryNodeStore(["a:9001"]), timeout=1.0)
    with pytest.raises(ValueError, match="must be at least 1") as exc_info:
        await cluster.connect(database="x", max_attempts=bad_value)
    assert f"got {bad_value}" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pool_and_cluster_max_attempts_messages_share_wording() -> None:
    """Both validators emit the same ``"must be at least 1"`` substring
    plus the same ``"got X"`` shape, so a regex like
    ``r"must be at least 1.*got"`` matches both."""
    pool_msg: str | None = None
    cluster_msg: str | None = None

    try:
        ConnectionPool(["a:9001"], max_attempts=0)
    except ValueError as e:
        pool_msg = str(e)

    cluster = ClusterClient(MemoryNodeStore(["a:9001"]), timeout=1.0)
    try:
        await cluster.connect(database="x", max_attempts=0)
    except ValueError as e:
        cluster_msg = str(e)

    assert pool_msg is not None and cluster_msg is not None
    # Same substring shape — operators can grep one regex for both.
    for msg in (pool_msg, cluster_msg):
        assert "must be at least 1" in msg
        assert "got 0" in msg
