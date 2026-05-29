"""When the cached leader self-confirms on the fast path, the redirect
policy must still be re-validated, or a tightened allowlist would not
take effect until the next leader flip invalidated the cache."""

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
    """A redirect_policy tightened after caching must be honoured on the
    next fast-path probe even when the cached node self-confirms."""
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

    def _strict_policy(addr: str) -> bool:
        return addr != "node-a:9001"

    cluster._redirect_policy = _strict_policy

    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()
