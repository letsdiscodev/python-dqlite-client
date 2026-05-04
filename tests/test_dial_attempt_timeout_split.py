"""``ClusterClient`` and ``DqliteConnection`` accept separate
``dial_timeout`` (TCP-establish budget) and ``attempt_timeout``
(per-attempt envelope) kwargs. Mirrors go-dqlite's
``Config.DialTimeout`` / ``Config.AttemptTimeout`` and the
``connector.go:357-360`` nested-envelope shape.

Both default to ``timeout`` so existing callers see no change.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import connect, create_pool
from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore

# ---------------------------------------------------------------- defaults


def test_cluster_defaults_dial_and_attempt_to_timeout() -> None:
    """When ``dial_timeout`` and ``attempt_timeout`` are not
    explicitly passed, both must default to ``timeout``."""
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
    """Setting only one of the two must leave the other defaulting
    to ``timeout``."""
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


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("name", ["dial_timeout", "attempt_timeout"])
@pytest.mark.parametrize("bad_value", [0, -1.0, True])
def test_cluster_split_rejects_zero_negative_or_bool(name: str, bad_value: object) -> None:
    """``_validate_timeout`` rejects 0 / negative / bool values; the
    new kwargs must apply the same rule."""
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


# ---------------------------------------------------------------- forwarding


def test_cluster_from_addresses_forwards_split_kwargs() -> None:
    cluster = ClusterClient.from_addresses(
        ["localhost:9001"], timeout=10.0, dial_timeout=0.5, attempt_timeout=2.0
    )
    assert cluster._dial_timeout == 0.5
    assert cluster._attempt_timeout == 2.0


@pytest.mark.asyncio
async def test_create_pool_forwards_split_to_cluster_and_connections() -> None:
    """create_pool builds a ClusterClient with the split kwargs (when
    no externally-owned cluster is supplied), and pooled connections
    inherit the split too."""
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


# ---------------------------------------------------------------- semantics


@pytest.mark.asyncio
async def test_cluster_query_leader_uses_dial_timeout_for_dial() -> None:
    """``_query_leader`` must call ``open_connection_with_keepalive``
    with ``self._dial_timeout`` for the TCP-establish budget — NOT
    ``self._timeout``. We patch the underlying connection-open call
    and assert the timeout argument that ``wait_for`` was given."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=10.0, dial_timeout=0.5, attempt_timeout=2.0)

    async def slow_open(*args: object, **kwargs: object) -> tuple[object, object]:
        # Sleep longer than dial_timeout but shorter than timeout; if
        # the dial uses ``self._timeout``, the test will not catch a
        # regression. Sleep > dial_timeout = 0.5 → wait_for raises.
        await asyncio.sleep(2.0)
        return MagicMock(), MagicMock()

    with patch(
        "dqliteclient.cluster.open_connection_with_keepalive",
        side_effect=slow_open,
    ):
        # ``_query_leader`` returns ``None`` on OSError-family
        # (which subsumes TimeoutError on cancellation). The test
        # passes if it returns within the dial_timeout budget, NOT
        # after the full ``timeout=10.0``.
        start = asyncio.get_running_loop().time()
        result = await cluster._query_leader("localhost:9001")
        elapsed = asyncio.get_running_loop().time() - start

    assert result is None
    assert elapsed < 1.5, (
        f"expected query_leader to give up within ~dial_timeout (0.5s); "
        f"actually took {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_dqlite_connection_dial_timeout_used_for_open() -> None:
    """``DqliteConnection._connect_impl`` must use ``dial_timeout``
    for ``open_connection_with_keepalive``, not ``timeout``."""
    conn = DqliteConnection("localhost:9001", timeout=10.0, dial_timeout=0.3, attempt_timeout=2.0)

    async def slow_open(*args: object, **kwargs: object) -> tuple[object, object]:
        await asyncio.sleep(2.0)
        return MagicMock(), MagicMock()

    with patch(
        "dqliteclient.connection.open_connection_with_keepalive",
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
    """``attempt_timeout`` wraps the entire dial + handshake
    envelope. If the dial succeeds quickly but the handshake stalls
    forever, the envelope must fire."""
    conn = DqliteConnection("localhost:9001", timeout=10.0, dial_timeout=5.0, attempt_timeout=0.3)

    # Dial succeeds instantly; handshake stalls forever. The
    # attempt-timeout envelope must trip even though the dial
    # already completed.
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
            "dqliteclient.connection.open_connection_with_keepalive",
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


# ---------------------------------------------------------------- top-level connect


@pytest.mark.asyncio
async def test_top_level_connect_forwards_split_to_dqlite_connection() -> None:
    """``dqliteclient.connect(...)`` must forward ``dial_timeout``
    and ``attempt_timeout`` to ``DqliteConnection``."""

    captured: dict[str, object] = {}
    real_init = DqliteConnection.__init__

    def capture_init(self: DqliteConnection, *args, **kwargs):
        captured.update(kwargs)
        real_init(self, *args, **kwargs)

    async def fast_connect(self: DqliteConnection) -> None:
        return None  # no-op

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


# ---------------------------------------------------------------- admin path


@pytest.mark.asyncio
async def test_admin_connection_uses_dial_timeout() -> None:
    """``_admin_connection`` must use ``dial_timeout`` for the
    TCP-establish call."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=10.0, dial_timeout=0.3, attempt_timeout=2.0)

    async def slow_open(*args: object, **kwargs: object) -> tuple[object, object]:
        await asyncio.sleep(2.0)
        return MagicMock(), MagicMock()

    with patch(
        "dqliteclient.cluster.open_connection_with_keepalive",
        side_effect=slow_open,
    ):
        start = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            async with cluster._admin_connection("localhost:9001") as _proto:
                pass
        elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 1.0, (
        f"expected admin connection to give up within ~dial_timeout "
        f"(0.3s); actually took {elapsed:.3f}s"
    )
