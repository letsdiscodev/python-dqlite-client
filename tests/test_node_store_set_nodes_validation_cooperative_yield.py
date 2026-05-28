"""Pin: the async ``set_nodes`` paths run their per-node
``parse_address`` validation/dedup walk with a cooperative
``asyncio.sleep(0)`` yield every ``_NODE_VALIDATE_YIELD_EVERY``
entries, so a large input (up to the 10_000-node ``_MAX_NODE_COUNT``
cap) does not monopolise the event loop.

The default validation calls ``parse_address`` + ``ipaddress.ip_address``
per node — ~tens of microseconds per call on commodity hardware,
hundreds of milliseconds total at the cap. This mirrors the discipline
already applied to the structurally identical
``ClusterClient.cluster_info`` redirect-policy filter (see
``test_cluster_info_filter_cooperative_yield.py``): same per-node cost,
same cap. ``YamlNodeStore.set_nodes`` already offloads the cheaper YAML
serialise + fsync ritual to a worker thread but left the costlier
validation pre-pass inline on the loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dqliteclient.node_store import (
    _NODE_VALIDATE_YIELD_EVERY,
    MemoryNodeStore,
    NodeInfo,
    YamlNodeStore,
    _validate_and_normalise_nodes,
)
from dqlitewire import NodeRole


def _make_nodes(n: int) -> list[NodeInfo]:
    """``n`` distinct, valid ``NodeInfo`` with canonical IPv4 addresses."""
    return [
        NodeInfo(
            node_id=i + 1,
            address=f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}:9001",
            role=NodeRole.VOTER,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_memory_set_nodes_yields_on_large_input() -> None:
    store = MemoryNodeStore(["10.255.255.255:9001"])
    nodes = _make_nodes(2000)

    sleep_zero_calls = 0
    original_sleep = asyncio.sleep

    async def counting_sleep(delay: float, *args: object, **kwargs: object) -> None:
        nonlocal sleep_zero_calls
        if delay == 0:
            sleep_zero_calls += 1
        await original_sleep(delay, *args, **kwargs)

    from unittest.mock import patch

    with patch("dqliteclient.node_store.asyncio.sleep", new=counting_sleep):
        await store.set_nodes(nodes)

    expected_min = (2000 // _NODE_VALIDATE_YIELD_EVERY) - 1
    assert sleep_zero_calls >= expected_min, (
        f"MemoryNodeStore.set_nodes made only {sleep_zero_calls} sleep(0) calls; "
        f"expected at least {expected_min} for N=2000 / "
        f"yield-every={_NODE_VALIDATE_YIELD_EVERY}"
    )
    assert len(await store.get_nodes()) == 2000


@pytest.mark.asyncio
async def test_yaml_set_nodes_yields_on_large_input(tmp_path: Path) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    nodes = _make_nodes(2000)

    sleep_zero_calls = 0
    original_sleep = asyncio.sleep

    async def counting_sleep(delay: float, *args: object, **kwargs: object) -> None:
        nonlocal sleep_zero_calls
        if delay == 0:
            sleep_zero_calls += 1
        await original_sleep(delay, *args, **kwargs)

    from unittest.mock import patch

    with patch("dqliteclient.node_store.asyncio.sleep", new=counting_sleep):
        await store.set_nodes(nodes)

    expected_min = (2000 // _NODE_VALIDATE_YIELD_EVERY) - 1
    assert sleep_zero_calls >= expected_min, (
        f"YamlNodeStore.set_nodes made only {sleep_zero_calls} sleep(0) calls; "
        f"expected at least {expected_min} for N=2000 / "
        f"yield-every={_NODE_VALIDATE_YIELD_EVERY}"
    )
    assert len(await store.get_nodes()) == 2000


@pytest.mark.asyncio
async def test_memory_set_nodes_small_input_pays_zero_yields() -> None:
    """The 1-9-node common case must call ``asyncio.sleep(0)`` zero
    times so it pays no scheduler overhead."""
    store = MemoryNodeStore(["10.255.255.255:9001"])
    nodes = _make_nodes(5)

    sleep_zero_calls = 0
    original_sleep = asyncio.sleep

    async def counting_sleep(delay: float, *args: object, **kwargs: object) -> None:
        nonlocal sleep_zero_calls
        if delay == 0:
            sleep_zero_calls += 1
        await original_sleep(delay, *args, **kwargs)

    from unittest.mock import patch

    with patch("dqliteclient.node_store.asyncio.sleep", new=counting_sleep):
        await store.set_nodes(nodes)

    assert sleep_zero_calls == 0, (
        f"small set_nodes made {sleep_zero_calls} sleep(0) calls; expected 0"
    )


@pytest.mark.asyncio
async def test_async_validation_byte_identical_to_sync() -> None:
    """The async-yielding validator must produce output and exceptions
    byte-identical to the synchronous helper."""
    from dqliteclient.node_store import _validate_and_normalise_nodes_async

    # Mix of plain, whitespace-laden, and duplicate-by-canonical entries
    # to exercise strip + dedup ordering.
    nodes = [
        NodeInfo(node_id=1, address="  10.0.0.1:9001  ", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="10.0.0.2:9001", role=NodeRole.SPARE),
        NodeInfo(node_id=3, address="10.0.0.1:9001", role=NodeRole.STANDBY),  # dup of #1
        *_make_nodes(300),
    ]
    sync_out = _validate_and_normalise_nodes(nodes)
    async_out = await _validate_and_normalise_nodes_async(nodes)
    assert sync_out == async_out

    # Exception parity: non-str address -> TypeError on both.
    bad_type = [NodeInfo(node_id=1, address=123, role=NodeRole.VOTER)]  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _validate_and_normalise_nodes(bad_type)
    with pytest.raises(TypeError):
        await _validate_and_normalise_nodes_async(bad_type)

    # Exception parity: empty stripped address -> ValueError on both.
    bad_empty = [NodeInfo(node_id=1, address="   ", role=NodeRole.VOTER)]
    with pytest.raises(ValueError):
        _validate_and_normalise_nodes(bad_empty)
    with pytest.raises(ValueError):
        await _validate_and_normalise_nodes_async(bad_empty)
