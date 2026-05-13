"""Pin: every DEBUG ``logger.debug`` call inside
``ClusterClient._find_leader_impl`` and its cached fast-path that
interpolates a peer-controlled address goes through
``sanitize_for_log`` (LF-escaping) rather than
``_sanitize_display_text`` (LF-preserving).

The six call sites covered:

- cached fast-path, redirect verify-failed
- cached fast-path, no-leader-known
- cached fast-path, transport failure
- ``_probe_one`` timeout
- ``_probe_one`` per-node failure
- ``_probe_one`` no-leader-known

CWE-117 log injection — a hostile peer pushing an LF into a node
address would forge an extra log line in journald / syslog when the
operator turns on DEBUG verbosity during an incident response.

``MemoryNodeStore.set_nodes`` rejects malformed addresses at the
store boundary, so these tests patch ``_safe_node_snapshot`` and
inject a ``NodeInfo`` that bypasses ``set_nodes`` validation — the
threat model is a third-party ``NodeStore`` implementation that
does not validate, OR a server-returned redirect target that
contains LF.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterError,
    DqliteConnectionError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.node_store import NodeInfo as _StoreNodeInfo
from dqlitewire import NodeRole

_LF_ADDR = "evil.example.com:9001\nFORGED log row"


def _make_cluster_with_poisoned_node(addr: str) -> ClusterClient:
    """Build a cluster whose ``_safe_node_snapshot`` returns a single
    NodeInfo containing LF — bypasses the store-side validator since the
    threat model is wire-supplied / 3rd-party-store-supplied LF."""
    store = MemoryNodeStore(["127.0.0.1:9001"])
    cc = ClusterClient(store, timeout=0.5, attempt_timeout=0.05)
    poisoned = _StoreNodeInfo(node_id=1, address=addr, role=NodeRole.VOTER)
    cc._safe_node_snapshot = AsyncMock(return_value=(poisoned,))
    return cc


def _assert_no_raw_lf_in_debug_records(caplog: pytest.LogCaptureFixture) -> None:
    """Every DEBUG record from ``dqliteclient.cluster`` must be a single
    line — a raw LF in the message would split into multiple records on
    the SIEM-ingest side."""
    debug_records = [
        r for r in caplog.records if r.name == "dqliteclient.cluster" and r.levelno == logging.DEBUG
    ]
    assert debug_records, "expected at least one DEBUG record from dqliteclient.cluster"
    for r in debug_records:
        msg = r.getMessage()
        assert "\n" not in msg, (
            f"raw LF leaked into a DEBUG log record: {msg!r} "
            f"(the address must be wrapped in sanitize_for_log, not "
            f"_sanitize_display_text)"
        )


def test_probe_one_timeout_debug_log_strips_lf(caplog: pytest.LogCaptureFixture) -> None:
    """``_probe_one`` TimeoutError arm."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    cc = _make_cluster_with_poisoned_node(_LF_ADDR)

    async def hang(*_a: object, **_kw: object) -> str | None:
        await asyncio.sleep(10.0)
        return None

    with (
        patch.object(cc, "_query_leader", side_effect=hang),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    _assert_no_raw_lf_in_debug_records(caplog)


def test_probe_one_failure_debug_log_strips_lf(caplog: pytest.LogCaptureFixture) -> None:
    """``_probe_one`` per-node failure arm."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    cc = _make_cluster_with_poisoned_node(_LF_ADDR)

    with (
        patch.object(
            cc,
            "_query_leader",
            side_effect=DqliteConnectionError("boom"),
        ),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    _assert_no_raw_lf_in_debug_records(caplog)


def test_probe_one_no_leader_known_debug_log_strips_lf(caplog: pytest.LogCaptureFixture) -> None:
    """``_probe_one`` no-leader-known arm (``_query_leader`` returns
    ``None``)."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    cc = _make_cluster_with_poisoned_node(_LF_ADDR)

    with (
        patch.object(cc, "_query_leader", new=AsyncMock(return_value=None)),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    _assert_no_raw_lf_in_debug_records(caplog)


def test_cached_fast_path_no_leader_debug_log_strips_lf(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cached fast-path arm where the cached node returns
    no-leader-known. The cache itself is set directly so LF reaches the
    fast-path log site."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    cc = _make_cluster_with_poisoned_node(_LF_ADDR)
    cc._set_last_known_leader(_LF_ADDR)

    with (
        patch.object(cc, "_query_leader", new=AsyncMock(return_value=None)),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    _assert_no_raw_lf_in_debug_records(caplog)


def test_cached_fast_path_transport_failure_debug_log_strips_lf(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cached fast-path arm where the cached node's probe fails with a
    transport-class exception."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    cc = _make_cluster_with_poisoned_node(_LF_ADDR)
    cc._set_last_known_leader(_LF_ADDR)

    with (
        patch.object(
            cc,
            "_query_leader",
            side_effect=DqliteConnectionError("boom"),
        ),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    _assert_no_raw_lf_in_debug_records(caplog)


def test_cached_fast_path_redirect_verify_failed_debug_log_strips_lf(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cached fast-path arm where the cached node redirects to a
    different leader and the verify step fails."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    cached_lf = "cached.example.com:9001\nFORGED cached-row"
    redirect_lf = "redirect.example.com:9002\nFORGED redirect-row"
    cc = _make_cluster_with_poisoned_node(cached_lf)
    cc._set_last_known_leader(cached_lf)

    async def fake_query_leader(addr: str, **_kw: object) -> str | None:
        # First call (cached node) returns a different leader; second
        # call (verify on the redirect) raises so verification fails.
        if addr == cached_lf:
            return redirect_lf
        raise ProtocolError("verify failed")

    with (
        patch.object(cc, "_query_leader", side_effect=fake_query_leader),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    _assert_no_raw_lf_in_debug_records(caplog)
