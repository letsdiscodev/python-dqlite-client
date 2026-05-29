"""``find_leader(policy=...)`` honours the per-call override so an
audit-mode caller's strict policy isn't downgraded to the permissive
instance default on the inner find-leader sweep."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_per_call_policy_overrides_permissive_instance_default() -> None:
    """A strict per-call policy must gate the redirect even when the
    instance default is permissive."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)  # permissive default
    with (
        patch.object(cc, "_query_leader", new=AsyncMock(return_value="attacker.com:9001")),
        pytest.raises(ClusterPolicyError, match="rejected"),
    ):
        asyncio.run(cc.find_leader(policy=allowlist_policy(["10.0.0.1:9001"])))


def test_per_call_policy_overrides_strict_instance_default() -> None:
    """A per-call policy can widen a reject-everything instance default."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        redirect_policy=lambda _a: False,
    )
    with patch.object(cc, "_query_leader", new=AsyncMock(return_value="anywhere:9001")):
        result = asyncio.run(cc.find_leader(policy=lambda _a: True))
    assert result == "anywhere:9001"


def test_per_call_policy_applied_to_cached_redirect_arm() -> None:
    """The cached-redirect fast-path arm must also honour the per-call
    policy."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)
    cc._set_last_known_leader("10.0.0.1:9001")  # pre-seed so fast path is taken
    with (
        patch.object(cc, "_query_leader", new=AsyncMock(return_value="attacker.com:9001")),
        pytest.raises(ClusterPolicyError, match="rejected"),
    ):
        asyncio.run(cc.find_leader(policy=allowlist_policy(["10.0.0.1:9001"])))


def test_per_call_policy_propagates_through_cluster_info() -> None:
    """``cluster_info(policy=...)`` forwards the per-call policy into the
    inner ``find_leader`` call."""
    from unittest.mock import MagicMock

    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)

    captured: dict[str, object] = {}

    async def fake_find_leader(*, policy=None, **_kw):
        captured["policy"] = policy
        return "10.0.0.1:9001"

    cc.find_leader = fake_find_leader

    fake_proto = MagicMock()
    fake_proto.cluster = AsyncMock(return_value=[])
    fake_proto.get_leader = AsyncMock(return_value=(1, "10.0.0.1:9001"))
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    p = allowlist_policy(["10.0.0.1:9001"])
    asyncio.run(cc.cluster_info(policy=p))
    assert captured["policy"] is p


def test_single_flight_key_includes_policy() -> None:
    """Two concurrent callers with different per-call policies must not
    share one in-flight task, or one's audit gate is silently degraded."""

    async def run() -> None:
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(store, timeout=5.0)

        call_count = 0
        gate = asyncio.Event()

        async def probe(address: str, **kw: object) -> str | None:
            nonlocal call_count
            call_count += 1
            await gate.wait()  # hold so both callers overlap
            return "10.0.0.1:9001"

        with patch.object(cc, "_query_leader", side_effect=probe):
            t1 = asyncio.create_task(cc.find_leader(policy=lambda _a: True))
            t2 = asyncio.create_task(cc.find_leader(policy=lambda _a: True))
            for _ in range(20):
                await asyncio.sleep(0)
            # Two distinct closures -> two slots -> two probes.
            assert call_count == 2
            gate.set()
            await asyncio.gather(t1, t2)

    asyncio.run(run())
