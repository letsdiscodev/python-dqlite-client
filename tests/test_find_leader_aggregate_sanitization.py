"""``_find_leader_impl``'s three aggregate-error interpolation sites all
sanitise ``node.address`` so a hostile entry cannot inject log lines
through the user-facing ``ClusterError`` message. LF and TAB are
preserved (multi-line server diagnostics); CR / bidi / control go to ``?``."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _hostile_store(address: str) -> MemoryNodeStore:
    """Inject directly to bypass the store's own CR/LF validation; this
    exercises the find_leader-side sanitise guard, not the store-side one."""
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
    # CR is sanitised to ``?``; LF is preserved per the wire-layer contract.
    assert "\r" not in msg, f"CR leaked into aggregate error: {msg!r}"
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
