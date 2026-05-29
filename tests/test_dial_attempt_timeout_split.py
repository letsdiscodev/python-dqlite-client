"""Separate dial_timeout (TCP-establish) and attempt_timeout (per-attempt envelope)
kwargs, mirroring go-dqlite; both default to timeout so existing callers see no change."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import connect, create_pool
from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


def test_cluster_defaults_dial_and_attempt_to_timeout() -> None:
    """Unset dial_timeout/attempt_timeout both default to timeout."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=2.5)
    assert cluster._dial_timeout == 2.5
    assert cluster._attempt_timeout == 2.5


def test_cluster_explicit_dial_and_attempt_kwargs_stored() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=10.0, dial_timeout=0.5, attempt_timeout=2.0)
    assert cluster._timeout == 10.0
    assert cluster._dial_timeout == 0.5
    assert cluster._attempt_timeout == 2.0


def test_cluster_partial_split_uses_timeout_for_unset_one() -> None:
    """Setting only one leaves the other defaulting to timeout."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=10.0, dial_timeout=0.3)
    assert cluster._dial_timeout == 0.3
    assert cluster._attempt_timeout == 10.0


def test_dqlite_connection_defaults_match_timeout() -> None:
    conn = DqliteConnection("localhost:9001", timeout=2.5)
    assert conn._dial_timeout == 2.5
    assert conn._attempt_timeout == 2.5


def test_dqlite_connection_explicit_split_stored() -> None:
    conn = DqliteConnection("localhost:9001", timeout=10.0, dial_timeout=0.5, attempt_timeout=2.0)
    assert conn._timeout == 10.0
    assert conn._dial_timeout == 0.5
    assert conn._attempt_timeout == 2.0


@pytest.mark.parametrize("name", ["dial_timeout", "attempt_timeout"])
@pytest.mark.parametrize("bad_value", [0, -1.0, True])
def test_cluster_split_rejects_zero_negative_or_bool(name: str, bad_value: object) -> None:
    """The new kwargs reject 0/negative/bool like validate_timeout."""
    store = MemoryNodeStore(["localhost:9001"])
    kwargs = {name: bad_value}
    with pytest.raises((TypeError, ValueError)):
        ClusterClient(store, timeout=10.0, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["dial_timeout", "attempt_timeout"])
@pytest.mark.parametrize("bad_value", [0, -1.0, True])
def test_dqlite_connection_split_rejects_zero_negative_or_bool(
    name: str, bad_value: object
) -> None:
    kwargs = {name: bad_value}
    with pytest.raises((TypeError, ValueError)):
        DqliteConnection("localhost:9001", timeout=10.0, **kwargs)  # type: ignore[arg-type]


def test_cluster_from_addresses_forwards_split_kwargs() -> None:
    cluster = ClusterClient.from_addresses(
        ["localhost:9001"], timeout=10.0, dial_timeout=0.5, attempt_timeout=2.0
    )
    assert cluster._dial_timeout == 0.5
    assert cluster._attempt_timeout == 2.0


@pytest.mark.asyncio
async def test_create_pool_forwards_split_to_cluster_and_connections() -> None:
    """create_pool forwards the split kwargs to the auto-built cluster and to connections."""
    pool = await create_pool(
        addresses=["localhost:9001"],
        dial_timeout=0.5,
        attempt_timeout=2.0,
        min_size=0,
        max_size=1,
    )
    try:
        assert pool._cluster._dial_timeout == 0.5
        assert pool._cluster._attempt_timeout == 2.0
        assert pool._dial_timeout == 0.5
        assert pool._attempt_timeout == 2.0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_cluster_query_leader_uses_dial_timeout_for_dial() -> None:
    """_query_leader uses dial_timeout, not timeout, for the TCP-establish budget."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=10.0, dial_timeout=0.5, attempt_timeout=2.0)

    async def slow_open(*args: object, **kwargs: object) -> tuple[object, object]:
        # Sleep > dial_timeout (0.5) but < timeout (10) so a regression to
        # self._timeout would not trip wait_for here.
        await asyncio.sleep(2.0)
        return MagicMock(), MagicMock()

    with patch(
        "dqliteclient._dial.open_connection_with_keepalive",
        side_effect=slow_open,
    ):
        start = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await cluster._query_leader("localhost:9001")
        elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 1.5, (
        f"expected query_leader to give up within ~dial_timeout (0.5s); "
        f"actually took {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_dqlite_connection_dial_timeout_used_for_open() -> None:
    """_connect_impl uses dial_timeout, not timeout, for the open call."""
    conn = DqliteConnection("localhost:9001", timeout=10.0, dial_timeout=0.3, attempt_timeout=2.0)

    async def slow_open(*args: object, **kwargs: object) -> tuple[object, object]:
        await asyncio.sleep(2.0)
        return MagicMock(), MagicMock()

    with patch(
        "dqliteclient._dial.open_connection_with_keepalive",
        side_effect=slow_open,
    ):
        start = asyncio.get_running_loop().time()
        with pytest.raises(DqliteConnectionError, match="timed out"):
            await conn.connect()
        elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 1.5, (
        f"expected DqliteConnection.connect to give up within "
        f"~dial_timeout (0.3s); actually took {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_dqlite_connection_attempt_timeout_envelope_wraps_handshake() -> None:
    """attempt_timeout wraps dial + handshake: a fast dial then a stuck
    handshake must still trip the envelope."""
    conn = DqliteConnection("localhost:9001", timeout=10.0, dial_timeout=5.0, attempt_timeout=0.3)

    fake_reader = MagicMock()
    fake_writer = MagicMock()
    fake_writer.close = MagicMock()
    fake_writer.wait_closed = AsyncMock()

    async def fast_open(*args: object, **kwargs: object) -> tuple[object, object]:
        return fake_reader, fake_writer

    async def stuck_handshake(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(60.0)

    with (
        patch(
            "dqliteclient._dial.open_connection_with_keepalive",
            side_effect=fast_open,
        ),
        patch(
            "dqliteclient.connection.DqliteProtocol.handshake",
            side_effect=stuck_handshake,
        ),
    ):
        start = asyncio.get_running_loop().time()
        with pytest.raises(DqliteConnectionError, match="timed out"):
            await conn.connect()
        elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 1.0, (
        f"expected attempt_timeout (0.3s) to bound the handshake; actually took {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_top_level_connect_forwards_split_to_dqlite_connection() -> None:
    """connect() forwards dial_timeout and attempt_timeout to DqliteConnection."""

    captured: dict[str, object] = {}
    real_init = DqliteConnection.__init__

    def capture_init(self: DqliteConnection, *args, **kwargs):
        captured.update(kwargs)
        real_init(self, *args, **kwargs)

    async def fast_connect(self: DqliteConnection) -> None:
        return None

    with (
        patch.object(DqliteConnection, "__init__", capture_init),
        patch.object(DqliteConnection, "connect", fast_connect),
    ):
        await connect(
            "localhost:9001",
            timeout=10.0,
            dial_timeout=0.5,
            attempt_timeout=2.0,
        )

    assert captured.get("dial_timeout") == 0.5
    assert captured.get("attempt_timeout") == 2.0


@pytest.mark.asyncio
async def test_admin_connection_uses_dial_timeout() -> None:
    """open_admin_connection uses dial_timeout for the TCP-establish call."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=10.0, dial_timeout=0.3, attempt_timeout=2.0)

    async def slow_open(*args: object, **kwargs: object) -> tuple[object, object]:
        await asyncio.sleep(2.0)
        return MagicMock(), MagicMock()

    with patch(
        "dqliteclient._dial.open_connection_with_keepalive",
        side_effect=slow_open,
    ):
        start = asyncio.get_running_loop().time()
        # Dial timeout wraps as DqliteConnectionError for admin paths.
        with pytest.raises(DqliteConnectionError, match="timed out"):
            async with cluster.open_admin_connection("localhost:9001") as _proto:
                pass
        elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 1.0, (
        f"expected admin connection to give up within ~dial_timeout "
        f"(0.3s); actually took {elapsed:.3f}s"
    )
