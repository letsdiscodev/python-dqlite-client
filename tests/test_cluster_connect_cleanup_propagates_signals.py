"""Pin: ``ClusterClient.connect``'s try_connect cleanup arm
must NOT swallow unexpected exceptions raised by the
shielded ``conn.close()``. The wide
``contextlib.suppress(BaseException)`` catches a
programming-bug class like ``AttributeError`` that the
narrow canonical pattern (``suppress(asyncio.CancelledError)``
plus a transport-class except) deliberately lets propagate.

Mirrors the pool's discipline at
``pool.py:1018-1042``:

* ``CancelledError``: absorbed (asyncio re-delivers at next
  await; the bare ``raise`` re-delivers the original
  handshake exception).
* ``OSError`` / ``DqliteConnectionError``: caught and logged
  (transport-class teardown failures on a half-built conn
  are expected).
* anything else (KI / SystemExit / unexpected ``Exception``
  subclasses): propagate.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient


@pytest.mark.asyncio
async def test_connect_cleanup_arm_does_not_swallow_unexpected_exception() -> None:
    cluster = ClusterClient.from_addresses(["localhost:9001"], timeout=0.5)

    async def _fake_find_leader(**kwargs: object) -> str:
        return "localhost:9001"

    cluster.find_leader = _fake_find_leader  # type: ignore[method-assign]

    import dqliteclient.cluster as cluster_mod

    real_dc = cluster_mod.DqliteConnection

    class _StubDqliteConnection:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        async def connect(self) -> None:
            raise OSError("simulated handshake failure")

        async def close(self) -> None:
            # Programming-bug class — the wide BaseException
            # suppress would silently swallow this; the narrow
            # canonical pattern lets it propagate so the real
            # source of the bug surfaces.
            raise AttributeError("unexpected attribute access in close()")

    cluster_mod.DqliteConnection = _StubDqliteConnection  # type: ignore[assignment]

    try:
        with pytest.raises(AttributeError, match="unexpected attribute access"):
            await cluster.connect("default")
    finally:
        cluster_mod.DqliteConnection = real_dc  # type: ignore[assignment]
