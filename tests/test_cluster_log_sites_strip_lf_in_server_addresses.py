"""Pin: every ``logger.*`` call in ``cluster.py`` that interpolates a
server-controlled address goes through ``sanitize_for_log`` (which
escapes LF / Tab) rather than ``sanitize_server_text`` (which
preserves them for exception-message readability).

The split is documented in ``dqlitewire/messages/responses.py``:
``sanitize_server_text`` strips C0 / C1 / bidi overrides / invisible
characters but **deliberately preserves LF and Tab** so exception
messages remain readable for interactive debugging.
``sanitize_for_log`` adds LF / Tab escaping on top, intended for
logger records where a raw LF would split the record into multiple
lines on the way through journald / syslog / Docker stdout.

Mixing them at a logger site is a CWE-117 log-injection hazard: a
hostile peer that returns a ``LeaderResponse.address`` /
``ServersResponse.NodeInfo.address`` containing ``\\n`` followed by
a forged log line attributable to dqlite would split the record on
the SIEM ingest side.

The four sites the audit identified:

- ``find_leader`` redirect-rejected DEBUG (cluster.py:545-547)
- ``find_leader`` redirect-verify-failed DEBUG (cluster.py:902-912)
- ``verify_redirect`` stale-hint DEBUG (cluster.py:1306-1310)
- ``cluster_info`` dropping-node WARNING (cluster.py:1601-1606) —
  highest exposure (WARNING level reaches SIEM)
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient import ClusterPolicyError, NodeInfo
from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole


@pytest.fixture
def cluster() -> ClusterClient:
    """Cluster with a single seed node and a tight allowlist so the
    redirect-policy rejection arm fires when the simulated server
    advises a peer outside the allowlist."""
    store = MemoryNodeStore(["127.0.0.1:9001"])
    return ClusterClient(
        store,
        redirect_policy=allowlist_policy(["127.0.0.1:9001"]),
    )


@pytest.mark.asyncio
async def test_check_redirect_logs_strip_lf_in_server_address(
    cluster: ClusterClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Site 1 — cluster.py:545-547. The DEBUG arm fires when the user-
    supplied ``redirect_policy`` rejects a server-advised redirect
    target. The address comes straight from the wire."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    address_with_lf = "evil.example.com:9001\nFORGED log row"

    with pytest.raises(ClusterPolicyError):
        cluster._check_redirect(address_with_lf)

    rec = next(
        (r for r in caplog.records if "redirect rejected by policy" in r.getMessage()),
        None,
    )
    assert rec is not None
    msg = rec.getMessage()
    assert "\n" not in msg, (
        f"server-controlled address must be sanitised before logging — "
        f"raw LF leaked into a logger record: {msg!r}"
    )


@pytest.mark.asyncio
async def test_verify_redirect_logs_strip_lf_in_reported_address(
    cluster: ClusterClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Site 3 — cluster.py:1306-1310. The stale-hint DEBUG fires when
    ``_verify_redirect`` dials the hinted peer and the peer reports
    a different leader address. ``reported`` is server-supplied."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    address_with_lf = "evil.example.com:9001\nFORGED leader-row"

    # Stub _query_leader to return the LF-poisoned address. The
    # _verify_redirect path dials the hint, queries leader, gets back
    # a different reported address, logs the discrepancy.
    cluster._query_leader = AsyncMock(return_value=address_with_lf)
    cluster._check_redirect = MagicMock(return_value=None)

    await cluster._verify_redirect("127.0.0.1:9001")

    rec = next(
        (r for r in caplog.records if "verify_redirect:" in r.getMessage()),
        None,
    )
    assert rec is not None
    msg = rec.getMessage()
    assert "\n" not in msg, (
        f"server-reported address must be sanitised before logging — "
        f"raw LF leaked into a logger record: {msg!r}"
    )


@pytest.mark.asyncio
async def test_cluster_info_warning_strips_lf_in_dropped_node_address(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Site 4 — cluster.py:1601-1606. WARNING level reaches SIEM /
    journald / syslog, so this is the highest-exposure log site.
    ``node.address`` per-entry is server-supplied (every entry of
    ``ServersResponse``).

    Drive the WARNING arm by mocking ``open_admin_connection`` to
    return a protocol whose ``cluster()`` reply includes a node with
    LF in its address, and a policy that rejects every node so the
    drop-and-log path fires."""
    import contextlib

    caplog.set_level(logging.WARNING, logger="dqliteclient.cluster")
    store = MemoryNodeStore(["127.0.0.1:9001"])
    cluster = ClusterClient(store, redirect_policy=lambda _: False)

    address_with_lf = "evil.example.com:9001\nFORGED warning-row"
    poisoned = NodeInfo(node_id=42, address=address_with_lf, role=NodeRole.VOTER)

    protocol = MagicMock()
    protocol.cluster = AsyncMock(return_value=[poisoned])

    @contextlib.asynccontextmanager
    async def fake_open_admin_connection(*_args: object, **_kwargs: object):
        yield protocol

    cluster.find_leader = AsyncMock(return_value="127.0.0.1:9001")
    cluster.open_admin_connection = fake_open_admin_connection

    await cluster.cluster_info()

    rec = next(
        (r for r in caplog.records if "cluster_info: dropping node" in r.getMessage()),
        None,
    )
    assert rec is not None
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert "\n" not in msg, (
        f"server-supplied node.address must be sanitised before logging at "
        f"WARNING — raw LF leaked into a logger record reaching SIEM: {msg!r}"
    )
