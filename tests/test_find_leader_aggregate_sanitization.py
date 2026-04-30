"""Pin: ``ClusterClient._find_leader_impl``'s three
aggregate-error interpolation sites all run
``_sanitize_display_text(node.address)`` so a hostile
allowlist entry (or a redirected node carrying CR /
U+2028 / U+2029 / control chars) cannot inject log lines
through the user-facing ``ClusterError`` message.

Per the wire-layer ``_sanitize_server_text`` contract,
LF and TAB are deliberately preserved so multi-line
server diagnostics still render. The sanitiser replaces
CR, U+2028, U+2029, and other control / bidi / invisible
characters with ``?``.

Cycle 22 added the sanitisation but no test pins the
contract. A refactor that consolidates the three error
lines into a helper and forgets one of the sanitise calls
would silently re-introduce the log-injection vector.

Three sites (one per error branch):

1. ``no leader known yet`` arm (``_query_leader`` returned None).
2. ``timed out`` arm (``TimeoutError``).
3. transport-error arm (``DqliteConnectionError`` / ``OSError`` /
   ``OperationalError``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _hostile_store(address: str) -> MemoryNodeStore:
    """Bypass the store's own validation (which strips CR / LF)
    by injecting directly. The test exercises the find_leader-side
    sanitise guard, not the store-side one."""
    store = MemoryNodeStore()
    object.__setattr__(
        store,
        "_nodes",
        (NodeInfo(node_id=1, address=address, role=NodeRole.VOTER),),
    )
    return store


@pytest.mark.asyncio
async def test_aggregate_error_sanitises_no_leader_known_branch() -> None:
    hostile = "evil:9001\r\nINJECTED-LOG-LINE"
    cc = ClusterClient(_hostile_store(hostile), timeout=0.1)
    cc._query_leader = AsyncMock(return_value=None)

    with pytest.raises(ClusterError) as exc_info:
        await cc.find_leader()

    msg = str(exc_info.value)
    # CR is sanitized (replaced with ``?``); LF is deliberately
    # preserved per the wire-layer contract (multi-line server
    # diagnostics).
    assert "\r" not in msg, f"CR leaked into aggregate error: {msg!r}"
    # The original hostile text is retained (just CR-escaped) so
    # operators can still triage.
    assert "INJECTED-LOG-LINE" in msg


@pytest.mark.asyncio
async def test_aggregate_error_sanitises_timeout_branch() -> None:
    hostile = "evil:9002\r\nINJECTED-FROM-TIMEOUT"
    cc = ClusterClient(_hostile_store(hostile), timeout=0.1)

    async def _raise_timeout(*a: object, **kw: object) -> None:
        raise TimeoutError()

    cc._query_leader = AsyncMock(side_effect=_raise_timeout)

    with pytest.raises(ClusterError) as exc_info:
        await cc.find_leader()

    msg = str(exc_info.value)
    assert "\r" not in msg
    assert "INJECTED-FROM-TIMEOUT" in msg


@pytest.mark.asyncio
async def test_aggregate_error_sanitises_transport_error_branch() -> None:
    hostile = "evil:9003\r\nINJECTED-FROM-TRANSPORT"
    cc = ClusterClient(_hostile_store(hostile), timeout=0.1)

    async def _raise_transport(*a: object, **kw: object) -> None:
        raise DqliteConnectionError("connection refused")

    cc._query_leader = AsyncMock(side_effect=_raise_transport)

    with pytest.raises(ClusterError) as exc_info:
        await cc.find_leader()

    msg = str(exc_info.value)
    assert "\r" not in msg
    assert "INJECTED-FROM-TRANSPORT" in msg
