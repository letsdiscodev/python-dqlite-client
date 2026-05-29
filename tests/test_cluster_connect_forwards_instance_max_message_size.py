"""connect() falls back to self._max_message_size when the per-call kwarg is
omitted; previously it forwarded None verbatim, silently dropping the
constructor-supplied value and bypassing the operator's DoS-hardening cap."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def _stub_open_connection() -> tuple[AsyncMock, MagicMock]:
    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()
    return mock_reader, mock_writer


@pytest.mark.asyncio
async def test_connect_forwards_instance_max_message_size_when_kwarg_omitted() -> None:
    """Construction-time max_message_size is honored when no per-call override."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=1.0, max_message_size=128 * 1024 * 1024)

    captured: dict[str, object] = {}

    original_init = DqliteConnection.__init__

    def _spy_init(self: DqliteConnection, *args: object, **kwargs: object) -> None:
        captured["max_message_size"] = kwargs.get("max_message_size")
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    mock_reader, mock_writer = _stub_open_connection()

    async def _fake_find_leader(self: ClusterClient, **_: object) -> str:
        return "localhost:9001"

    async def _fake_connect(self: DqliteConnection) -> None:
        return None

    with (
        patch.object(ClusterClient, "find_leader", _fake_find_leader),
        patch.object(DqliteConnection, "__init__", _spy_init),
        patch.object(DqliteConnection, "connect", _fake_connect),
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
    ):
        await cluster.connect()

    assert captured["max_message_size"] == 128 * 1024 * 1024


@pytest.mark.asyncio
async def test_connect_per_call_override_wins_over_instance_default() -> None:
    """Per-call max_message_size kwarg wins over the instance value."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=1.0, max_message_size=128 * 1024 * 1024)

    captured: dict[str, object] = {}

    original_init = DqliteConnection.__init__

    def _spy_init(self: DqliteConnection, *args: object, **kwargs: object) -> None:
        captured["max_message_size"] = kwargs.get("max_message_size")
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    mock_reader, mock_writer = _stub_open_connection()

    async def _fake_find_leader(self: ClusterClient, **_: object) -> str:
        return "localhost:9001"

    async def _fake_connect(self: DqliteConnection) -> None:
        return None

    with (
        patch.object(ClusterClient, "find_leader", _fake_find_leader),
        patch.object(DqliteConnection, "__init__", _spy_init),
        patch.object(DqliteConnection, "connect", _fake_connect),
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
    ):
        await cluster.connect(max_message_size=4 * 1024 * 1024)

    assert captured["max_message_size"] == 4 * 1024 * 1024


@pytest.mark.asyncio
async def test_connect_with_no_instance_default_and_no_kwarg_passes_none() -> None:
    """Unset construction value and omitted kwarg propagate None (no cluster-layer
    default injection); DqliteConnection resolves it to the wire default."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=1.0)  # no max_message_size

    captured: dict[str, object] = {}

    original_init = DqliteConnection.__init__

    def _spy_init(self: DqliteConnection, *args: object, **kwargs: object) -> None:
        captured["max_message_size"] = kwargs.get("max_message_size")
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    mock_reader, mock_writer = _stub_open_connection()

    async def _fake_find_leader(self: ClusterClient, **_: object) -> str:
        return "localhost:9001"

    async def _fake_connect(self: DqliteConnection) -> None:
        return None

    with (
        patch.object(ClusterClient, "find_leader", _fake_find_leader),
        patch.object(DqliteConnection, "__init__", _spy_init),
        patch.object(DqliteConnection, "connect", _fake_connect),
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
    ):
        await cluster.connect()

    assert captured["max_message_size"] is None
