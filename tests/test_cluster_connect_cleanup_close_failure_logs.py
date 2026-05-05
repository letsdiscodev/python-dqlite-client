"""``ClusterClient.connect`` cleanup-close failure debug-log arm.

After a handshake exception, the cleanup ``await
asyncio.shield(conn.close())`` may itself raise OSError /
DqliteConnectionError. The logger.debug emit at
``cluster.py:1365-1368`` is the only forensic trail when this
happens; pin it so a refactor cannot drop the catch and surface
OSError instead of the real handshake error.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ProtocolError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_connect_cleanup_close_failure_debug_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A handshake failure followed by a close-side OSError must
    propagate the original error and emit a DEBUG log on the close
    failure."""
    cluster = ClusterClient(MemoryNodeStore(["h:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="h:9001")
    cluster._get_last_known_leader = MagicMock(return_value="h:9001")

    # Simulate: open_connection succeeds, handshake raises
    # ProtocolError, the cleanup conn.close() then raises OSError.
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
        # The ProtocolError → DqliteConnectionError rewrap on connect
        # has the original cause chained.
        pytest.raises(ProtocolError),
    ):
        await cluster.connect(database="default", max_attempts=1)

    # Debug log emitted on the close-side OSError.
    assert any("cleanup: conn.close() failed" in r.message for r in caplog.records)
