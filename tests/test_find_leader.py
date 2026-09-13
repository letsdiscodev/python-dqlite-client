"""ClusterClient.find_leader: ordering, aggregate diagnostics, malformed replies, probe bounds."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_role_bucket_sort_is_stable_within_same_role() -> None:
    """The role-int sort is stable, preserving same-role nodes' relative
    order so a non-stable regression would lose shuffle avoidance."""
    nodes_in_input_order = [
        NodeInfo(1, "sb_a:9001", NodeRole.STANDBY),
        NodeInfo(2, "sb_b:9001", NodeRole.STANDBY),
        NodeInfo(3, "voter:9001", NodeRole.VOTER),
        NodeInfo(4, "sb_c:9001", NodeRole.STANDBY),
    ]
    out = sorted(nodes_in_input_order, key=lambda n: int(n.role))
    # VOTER first, then STANDBYs in input order (stable sort).
    assert [n.address for n in out] == [
        "voter:9001",
        "sb_a:9001",
        "sb_b:9001",
        "sb_c:9001",
    ]


def _build_cluster_with_mixed_roles() -> ClusterClient:
    addresses = [
        "v1:9001",  # VOTER
        "sb1:9001",
        "sb2:9001",
        "sb3:9001",  # 3x STANDBY
        "sp1:9001",
        "sp2:9001",
        "sp3:9001",  # 3x SPARE
    ]
    store = MemoryNodeStore(addresses)
    # MemoryNodeStore defaults every node to VOTER; we want a mixed distribution.
    role_map = {
        "v1:9001": NodeRole.VOTER,
        "sb1:9001": NodeRole.STANDBY,
        "sb2:9001": NodeRole.STANDBY,
        "sb3:9001": NodeRole.STANDBY,
        "sp1:9001": NodeRole.SPARE,
        "sp2:9001": NodeRole.SPARE,
        "sp3:9001": NodeRole.SPARE,
    }
    # NodeInfo is frozen; rebuild the internal tuple.
    rebuilt = tuple(
        NodeInfo(node_id=i + 1, address=addr, role=role_map[addr])
        for i, addr in enumerate(addresses)
    )
    store._nodes = rebuilt
    return ClusterClient(store, concurrent_leader_conns=1, attempt_timeout=2.0)


@pytest.mark.asyncio
async def test_find_leader_probes_standby_before_spare() -> None:
    """With ``concurrent_leader_conns=1`` probing is sequential, making the sort observable."""
    cluster = _build_cluster_with_mixed_roles()

    probe_order: list[str] = []

    async def _record(addr: str, **_kw: object) -> str | None:
        probe_order.append(addr)
        return None  # no leader known — drive the full sweep

    with (
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_record)),
        contextlib.suppress(Exception),  # all-no-leader → ClusterError; we want probe order
    ):
        await cluster.find_leader()

    sb_positions = [i for i, a in enumerate(probe_order) if a.startswith("sb")]
    sp_positions = [i for i, a in enumerate(probe_order) if a.startswith("sp")]
    v_positions = [i for i, a in enumerate(probe_order) if a.startswith("v")]

    if v_positions and sb_positions:
        assert max(v_positions) < min(sb_positions), (
            f"VOTER must precede STANDBY: probe order = {probe_order}"
        )
    if sb_positions and sp_positions:
        assert max(sb_positions) < min(sp_positions), (
            f"All STANDBY must precede any SPARE: probe order = {probe_order}"
        )


def _hand_build_leader_response_bytes(node_id: int, address: str) -> bytes:
    """Build LeaderResponse wire bytes bypassing the constructor's atomicity check."""
    from dqlitewire.constants import ResponseType
    from dqlitewire.messages.base import Header
    from dqlitewire.types import encode_text, encode_uint64

    body = encode_uint64(node_id) + encode_text(address)
    header = Header(size_words=len(body) // 8, msg_type=ResponseType.LEADER, schema=0)
    return header.encode() + body


@pytest.mark.parametrize(
    "node_id,address_str",
    [
        (1, ""),
        (0, "peer:9000"),  # mirror arm — must also raise
    ],
)
@pytest.mark.asyncio
async def test_query_leader_rejects_both_wire_illegal_shapes(
    node_id: int, address_str: str
) -> None:
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=1.0)

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    from dqlitewire.messages import WelcomeResponse

    # The wire check only rejects ``(0, non-empty)``, so only that arm needs hand-built bytes.
    if node_id == 0 and address_str:
        leader_bytes = _hand_build_leader_response_bytes(node_id, address_str)
    else:
        from dqlitewire.messages import LeaderResponse

        leader_bytes = LeaderResponse(node_id=node_id, address=address_str).encode()

    mock_reader.read.side_effect = [
        WelcomeResponse(heartbeat_timeout=15000).encode(),
        leader_bytes,
    ]

    with (
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
        # ProtocolError propagates through find_leader as the retry loop's ClusterError.
        pytest.raises(ClusterError),
    ):
        await client.find_leader()


class _SlowNodeStore:
    """A NodeStore whose ``get_nodes`` blocks for 60 s."""

    async def get_nodes(self) -> Sequence[NodeInfo]:
        await asyncio.sleep(60)
        return ()

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        return None


async def test_slow_node_store_get_nodes_bounded_by_dial_timeout() -> None:
    store = _SlowNodeStore()
    client = ClusterClient(store, dial_timeout=0.1)
    start = time.monotonic()
    with pytest.raises((TimeoutError, DqliteConnectionError)):
        await client.find_leader()
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, (
        f"_safe_node_snapshot must bound the get_nodes call by dial_timeout; "
        f"elapsed {elapsed:.2f}s (expected ~0.1s)"
    )


def test_lf_in_str_e_does_not_survive_into_cluster_error_args0() -> None:
    """LF in peer text must reach ClusterError.args[0] escaped, not raw."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=0.5)

    hostile = DqliteConnectionError("first line\nforged second-line")
    with (
        patch.object(cc, "_query_leader", side_effect=hostile),
        pytest.raises(ClusterError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    raw_args0 = exc_info.value.args[0]
    assert "\n" not in raw_args0, (
        f"raw LF leaked into ClusterError.args[0]; downstream "
        f"logger.exception(ce) would split the record into multiple log "
        f"lines. Got: {raw_args0!r}"
    )
    assert "\\n" in raw_args0 or "forged second-line" in raw_args0


def test_tab_in_str_e_does_not_survive_into_cluster_error_args0() -> None:
    """Tab vector: confirm it's escaped too."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=0.5)

    hostile = DqliteConnectionError("first\tcolumn\tinjected")
    with (
        patch.object(cc, "_query_leader", side_effect=hostile),
        pytest.raises(ClusterError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    raw_args0 = exc_info.value.args[0]
    assert "\t" not in raw_args0, f"raw Tab leaked into ClusterError.args[0]: {raw_args0!r}"
