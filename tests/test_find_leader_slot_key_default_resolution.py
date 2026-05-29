"""The find_leader slot key uses the effective policy (after None ->
self._redirect_policy resolution), so a None caller and an explicit
instance-default caller share one task instead of running two probes."""

import asyncio
from unittest.mock import patch

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_slot_key_collapses_none_and_instance_default_policy() -> None:
    """``policy=None`` and ``policy=cc._redirect_policy`` callers must
    share one in-flight task."""

    async def run() -> None:
        store = MemoryNodeStore(["10.0.0.1:9001"])

        def instance_policy(_a: str) -> bool:
            return True

        cc = ClusterClient(store, timeout=5.0, redirect_policy=instance_policy)

        call_count = 0
        gate = asyncio.Event()

        async def probe(address: str, **_kw: object) -> str | None:
            nonlocal call_count
            call_count += 1
            await gate.wait()
            return "10.0.0.1:9001"

        with patch.object(cc, "_query_leader", side_effect=probe):
            t1 = asyncio.create_task(cc.find_leader(policy=None))
            t2 = asyncio.create_task(cc.find_leader(policy=cc._redirect_policy))
            for _ in range(20):
                await asyncio.sleep(0)
            assert call_count == 1, (
                f"Single-flight slot key should resolve None to "
                f"self._redirect_policy before keying so both callers "
                f"share one in-flight task; got {call_count} probes."
            )
            gate.set()
            await asyncio.gather(t1, t2)

    asyncio.run(run())
