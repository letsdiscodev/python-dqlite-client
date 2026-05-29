"""Every logger.* call in cluster.py interpolating a server-controlled address
uses sanitize_for_log (escapes LF/Tab), not sanitize_server_text (which preserves
them for exception readability). Mixing them is a CWE-117 log-injection hazard.
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
    """Single seed with a tight allowlist so the redirect-policy rejection arm fires."""
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
    """Site 1: _check_redirect DEBUG arm when redirect_policy rejects a wire address."""
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
    """Site 3: _verify_redirect stale-hint DEBUG when the peer reports a different
    (server-supplied) leader address."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    address_with_lf = "evil.example.com:9001\nFORGED leader-row"

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
    """Site 4: cluster_info dropping-node WARNING (highest exposure, reaches SIEM)
    with a server-supplied node.address; a reject-all policy fires the drop arm."""
    import contextlib

    caplog.set_level(logging.WARNING, logger="dqliteclient.cluster")
    store = MemoryNodeStore(["127.0.0.1:9001"])
    cluster = ClusterClient(store, redirect_policy=lambda _: False)

    address_with_lf = "evil.example.com:9001\nFORGED warning-row"
    poisoned = NodeInfo(node_id=42, address=address_with_lf, role=NodeRole.VOTER)

    protocol = MagicMock()
    protocol.cluster = AsyncMock(return_value=[poisoned])
    protocol.get_leader = AsyncMock(return_value=(1, "127.0.0.1:9001"))

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
