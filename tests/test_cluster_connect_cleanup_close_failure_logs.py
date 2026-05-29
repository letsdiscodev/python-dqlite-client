"""connect's cleanup conn.close() may itself raise after a handshake exception;
the DEBUG emit is the only forensic trail, and the original handshake error (not
the close OSError) must still propagate."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ProtocolError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_connect_cleanup_close_failure_debug_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Close-side OSError emits a DEBUG log; the original error still propagates."""
    cluster = ClusterClient(MemoryNodeStore(["h:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="h:9001")
    cluster._get_last_known_leader = MagicMock(return_value="h:9001")

    # handshake raises ProtocolError, then the cleanup conn.close() raises OSError.
    fake_streams = (AsyncMock(), AsyncMock())
    fake_streams[1].close = lambda: None
    fake_streams[1].wait_closed = AsyncMock()

    fake_conn = MagicMock()
    fake_conn.connect = AsyncMock(side_effect=ProtocolError("simulated handshake failure"))
    fake_conn.close = AsyncMock(side_effect=OSError("simulated close failure"))
    fake_conn.address = "h:9001"

    with (
        patch("dqliteclient.cluster.DqliteConnection", return_value=fake_conn),
        caplog.at_level("DEBUG"),
        pytest.raises(ProtocolError),
    ):
        await cluster.connect(database="default", max_attempts=1)

    assert any("cleanup: conn.close() failed" in r.message for r in caplog.records)
