"""Pin: the ``find_leader`` single-flight slot key uses the EFFECTIVE
policy (after default resolution), not the raw kwarg.

Two callers — one passing ``policy=None`` (which falls back to
``self._redirect_policy``) and one passing ``self._redirect_policy``
explicitly — describe the same effective probe and should share a
single in-flight task. Pre-fix, the raw ``policy`` value was used to
key the ``_find_leader_tasks`` slot, so ``(False, None)`` and
``(False, self._redirect_policy)`` were distinct entries: both callers
saw an empty slot, both created a task, and the second insertion
clobbered the first registration without cancelling it — TWO probes
ran in parallel.

The fix resolves ``policy`` to ``self._redirect_policy`` BEFORE
constructing the key and forwards the resolved value into
``_find_leader_impl``.
"""

import asyncio
from unittest.mock import patch

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_slot_key_collapses_none_and_instance_default_policy() -> None:
    """``find_leader()`` with ``policy=None`` and
    ``find_leader(policy=cc._redirect_policy)`` must share one
    in-flight task — the default-resolution collision is closed."""

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
            # One caller forwards None (the typical user-facing shape);
            # the other forwards the resolved instance default (the
            # internal re-dispatch shape).
            t1 = asyncio.create_task(cc.find_leader(policy=None))
            t2 = asyncio.create_task(cc.find_leader(policy=cc._redirect_policy))
            # Yield so both callers reach the slot lookup.
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
