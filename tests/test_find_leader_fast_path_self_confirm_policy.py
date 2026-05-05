"""When the cached leader self-confirms on the fast path, the
redirect policy MUST still be re-validated. Without this, a tightened
allowlist (set after the cache was populated) would not take effect
until the cache was independently invalidated by the next leader
flip — leaving a stale grace window where the old policy applied.

The cached node also was already redirect-policy-checked at the time
of caching, so the redirect arm at line 586 calls ``_check_redirect``
on the redirected target. The self-confirm arm is the symmetric case:
the cached address may now be outside the policy.
"""

from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError


def _query_leader_factory(
    leader_for: dict[str, str | None],
) -> Callable[..., object]:
    calls: list[str] = []

    async def _impl(address: str, **_kw: object) -> str | None:
        calls.append(address)
        return leader_for.get(address)

    _impl.calls = calls  # type: ignore[attr-defined]
    return _impl


@pytest.mark.asyncio
async def test_self_confirm_revalidates_redirect_policy() -> None:
    """A redirect_policy that is tightened after the cache was
    populated must be honoured on the next fast-path probe even
    when the cached node self-confirms."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(
        addresses,
        timeout=5.0,
    )

    leader_responses: dict[str, str | None] = {
        "node-a:9001": "node-a:9001",
        "node-b:9001": None,
    }
    impl = _query_leader_factory(leader_responses)
    cluster._query_leader = AsyncMock(side_effect=impl)

    # First sweep populates the cache with node-a (which self-confirms).
    leader = await cluster.find_leader()
    assert leader == "node-a:9001"
    assert cluster._get_last_known_leader() == "node-a:9001"

    # Tighten the policy: reject node-a entirely.
    def _strict_policy(addr: str) -> bool:
        return addr != "node-a:9001"

    cluster._redirect_policy = _strict_policy

    # Next fast-path probe must surface the policy rejection.
    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()
