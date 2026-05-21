"""``find_leader(policy=...)`` honours the per-call override.

Sibling admin methods ``leader_info`` and ``cluster_info`` already
accept a per-call ``policy=`` kwarg that overrides the instance-level
``redirect_policy``. ``find_leader`` is the gateway RPC for every
other admin method, so the per-call override must reach the
``_check_redirect`` gate on the find-leader codepath too — otherwise
an audit-mode caller passing ``cluster_info(policy=strict)`` is
silently downgraded to the permissive instance default for the
inner ``find_leader`` sweep.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_per_call_policy_overrides_permissive_instance_default() -> None:
    """An audit-mode caller passes ``policy=allowlist_policy([...])``
    to enforce a strict gate even though the instance default is
    permissive. The redirect target must be evaluated against the
    per-call policy, not ``self._redirect_policy``."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    # Instance default is permissive (None).
    cc = ClusterClient(store, timeout=5.0)
    with (
        patch.object(cc, "_query_leader", new=AsyncMock(return_value="attacker.com:9001")),
        pytest.raises(ClusterPolicyError, match="rejected"),
    ):
        asyncio.run(cc.find_leader(policy=allowlist_policy(["10.0.0.1:9001"])))


def test_per_call_policy_overrides_strict_instance_default() -> None:
    """The reverse direction: a caller can WIDEN the policy at call
    time by passing an always-True callable while the instance default
    rejects everything."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        redirect_policy=lambda _a: False,  # rejects everything
    )
    with patch.object(cc, "_query_leader", new=AsyncMock(return_value="anywhere:9001")):
        result = asyncio.run(cc.find_leader(policy=lambda _a: True))
    assert result == "anywhere:9001"


def test_per_call_policy_applied_to_cached_redirect_arm() -> None:
    """The cached-leader fast path arm at the ``cached redirected us
    elsewhere`` branch must also honour the per-call policy. Pre-seed
    the cache so the fast path is taken."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)
    cc._set_last_known_leader("10.0.0.1:9001")
    with (
        # Cached node redirects elsewhere.
        patch.object(cc, "_query_leader", new=AsyncMock(return_value="attacker.com:9001")),
        pytest.raises(ClusterPolicyError, match="rejected"),
    ):
        asyncio.run(cc.find_leader(policy=allowlist_policy(["10.0.0.1:9001"])))


def test_per_call_policy_propagates_through_cluster_info() -> None:
    """``cluster_info(policy=...)`` forwards the per-call policy into
    the inner ``find_leader`` call so the audit-mode policy gates the
    redirect arm too."""
    from unittest.mock import MagicMock

    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)

    # Capture what find_leader was called with.
    captured: dict[str, object] = {}

    async def fake_find_leader(*, policy=None, **_kw):
        captured["policy"] = policy
        return "10.0.0.1:9001"

    cc.find_leader = fake_find_leader

    fake_proto = MagicMock()
    fake_proto.cluster = AsyncMock(return_value=[])
    # Re-confirm leadership round-trip on the no-flip happy path.
    fake_proto.get_leader = AsyncMock(return_value=(1, "10.0.0.1:9001"))
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    p = allowlist_policy(["10.0.0.1:9001"])
    asyncio.run(cc.cluster_info(policy=p))
    assert captured["policy"] is p


def test_single_flight_key_includes_policy() -> None:
    """Two concurrent ``find_leader`` callers with DIFFERENT per-call
    policies must NOT share a single in-flight task — sharing would
    bind the caller to whichever policy the first acquirer happened
    to register, silently degrading their audit gate."""

    async def run() -> None:
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(store, timeout=5.0)

        call_count = 0
        gate = asyncio.Event()

        async def probe(address: str, **kw: object) -> str | None:
            nonlocal call_count
            call_count += 1
            # Hold so both concurrent callers overlap.
            await gate.wait()
            return "10.0.0.1:9001"

        with patch.object(cc, "_query_leader", side_effect=probe):
            t1 = asyncio.create_task(cc.find_leader(policy=lambda _a: True))
            t2 = asyncio.create_task(cc.find_leader(policy=lambda _a: True))
            # Yield enough times so both tasks have entered _query_leader.
            for _ in range(20):
                await asyncio.sleep(0)
            # Two distinct closures -> two single-flight slots -> two probes.
            assert call_count == 2
            gate.set()
            await asyncio.gather(t1, t2)

    asyncio.run(run())
