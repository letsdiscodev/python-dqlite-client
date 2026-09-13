"""ClusterClient.connect: attempt logging, cancellation, knob forwarding and per-call policy."""

from __future__ import annotations

import asyncio
import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import ClusterError, DqliteConnectionError, OperationalError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_connect_attempt_failed_debug_log_strips_lf_in_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An LF-carrying exception must not split the per-attempt DEBUG record."""
    addr_with_lf = "victim:9001"
    poisoned_message = "leader-msg\nFORGED log row"
    cluster = ClusterClient(MemoryNodeStore([addr_with_lf]))

    async def _exploding_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        raise DqliteConnectionError(poisoned_message)

    monkeypatch.setattr(cluster, "find_leader", _exploding_find_leader)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    with pytest.raises((ClusterError, DqliteConnectionError)):
        await cluster.connect()

    debug_records = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and "ClusterClient.connect attempt" in r.getMessage()
        and "failed" in r.getMessage()
    ]
    assert debug_records, "expected one or more 'attempt N/M failed' DEBUG records from connect()"
    for rec in debug_records:
        msg = rec.getMessage()
        assert "\n" not in msg, (
            f"per-attempt DEBUG log leaked raw LF from server exception message: {msg!r}"
        )


@pytest.mark.asyncio
async def test_try_connect_operationalerror_emits_attempt_log_and_propagates_unwrapped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
    )
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    async def _raise_operational_error(*args: object, **kwargs: object) -> object:
        # Non-leader-flip code: SQLITE_NOTFOUND (12).
        raise OperationalError("unknown database", code=12)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    with (
        patch.object(DqliteConnection, "connect", AsyncMock(side_effect=_raise_operational_error)),
        pytest.raises(OperationalError) as exc_info,
    ):
        await cluster.connect(max_attempts=1)

    # Propagates unwrapped — NOT rewrapped to DqliteConnectionError.
    assert exc_info.value.code == 12
    assert "unknown database" in str(exc_info.value)

    debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert any("ClusterClient.connect attempt" in r.message for r in debug_records), (
        "Expected per-attempt DEBUG breadcrumb. Got log records: "
        f"{[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_outer_cancel_before_first_attempt_does_not_log_breadcrumb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """timeout(0) lands the cancel before any try_connect; no breadcrumb fires."""
    cluster = ClusterClient(MemoryNodeStore(["10.0.0.1:9001"]))

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    with pytest.raises((TimeoutError, asyncio.CancelledError)):
        async with asyncio.timeout(0):
            await cluster.connect()

    # Pin on the specific breadcrumb phrases; bare "attempt" is too broad.
    breadcrumb_phrases = ("attempt failed", "Connection attempt")
    matching = [
        r for r in caplog.records if any(phrase in r.getMessage() for phrase in breadcrumb_phrases)
    ]
    assert not matching, (
        "Outer cancel-before-first-attempt produced per-attempt log records "
        f"({len(matching)}); a defensive CancelledError catch in try_connect "
        "is suppressing the structured-concurrency cancel contract. "
        "Records: " + " | ".join(r.getMessage() for r in matching)
    )


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


def test_connect_accepts_policy_kwarg() -> None:
    sig = inspect.signature(ClusterClient.connect)
    assert "policy" in sig.parameters
    assert sig.parameters["policy"].kind == inspect.Parameter.KEYWORD_ONLY
    # Default None so callers without an override fall back to the instance policy.
    assert sig.parameters["policy"].default is None
