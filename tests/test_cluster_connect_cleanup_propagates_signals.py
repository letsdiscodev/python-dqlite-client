"""connect's cleanup arm must not swallow unexpected exceptions from the shielded
conn.close(): only CancelledError and transport-class errors (OSError /
DqliteConnectionError) are absorbed; a programming-bug AttributeError propagates."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient


@pytest.mark.asyncio
async def test_connect_cleanup_arm_does_not_swallow_unexpected_exception() -> None:
    cluster = ClusterClient.from_addresses(["localhost:9001"], timeout=0.5)

    async def _fake_find_leader(**kwargs: object) -> str:
        return "localhost:9001"

    cluster.find_leader = _fake_find_leader

    import dqliteclient.cluster as cluster_mod

    real_dc = cluster_mod.DqliteConnection  # type: ignore[attr-defined]

    class _StubDqliteConnection:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        async def connect(self) -> None:
            raise OSError("simulated handshake failure")

        async def close(self) -> None:
            # Programming-bug class that must propagate, not be suppressed.
            raise AttributeError("unexpected attribute access in close()")

    cluster_mod.DqliteConnection = _StubDqliteConnection  # type: ignore[assignment,attr-defined]

    try:
        with pytest.raises(AttributeError, match="unexpected attribute access"):
            await cluster.connect("default")
    finally:
        cluster_mod.DqliteConnection = real_dc  # type: ignore[attr-defined]
