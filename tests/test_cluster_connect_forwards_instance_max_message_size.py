"""Pin: ``ClusterClient.connect()`` falls back to
``self._max_message_size`` when the per-call ``max_message_size`` kwarg
is not supplied.

Sibling sites (``_query_leader`` at cluster.py:1428 and an admin path
at cluster.py:2828) correctly consult ``self._max_message_size``. The
``connect()`` path took a kwarg-only ``max_message_size: int | None =
None`` and forwarded it verbatim into ``DqliteConnection(...)``, so a
caller doing ``ClusterClient(..., max_message_size=N)`` followed by
``await cluster.connect()`` (no per-call override) silently dropped
the constructor-supplied value. ``DqliteProtocol`` then resolved
``None`` to ``DEFAULT_MAX_MESSAGE_SIZE`` (the wire default ~64 MiB),
silently bypassing the operator's DoS-hardening cap or the operator's
"raise the cap to handle large rowsets" knob.

Same shape as the ``policy=`` precedent at cluster.py:744-756 ("per-
call overrides instance default; None falls back").
"""

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
    """Construction-time ``max_message_size`` is honored by ``connect()``
    when the caller doesn't supply a per-call override."""
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
    """Per-call ``max_message_size`` kwarg wins over the instance value
    — matches the ``policy=`` precedent at cluster.py:744-756."""
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
    """Regression: an unset construction-time value AND an omitted
    per-call kwarg result in ``None`` propagating to ``DqliteConnection``
    (which then resolves to the wire default). No silent default
    injection at the cluster layer."""
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
