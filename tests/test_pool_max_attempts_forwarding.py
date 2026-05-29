"""``ConnectionPool`` and ``create_pool`` forward ``max_attempts`` to
``ClusterClient.connect()`` so pool users can tune connect-retry without a custom
``ClusterClient``."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient import create_pool
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_connection_pool_forwards_max_attempts() -> None:
    """``ConnectionPool(..., max_attempts=N)`` forwards N to
    ``cluster.connect``."""
    pool = ConnectionPool(["a:9001", "b:9002"], max_attempts=7)
    fake_connect = AsyncMock(return_value=AsyncMock())
    pool._cluster.connect = fake_connect

    await pool._create_connection()

    fake_connect.assert_awaited_once()
    assert fake_connect.await_args is not None
    call_kwargs = fake_connect.await_args.kwargs
    assert call_kwargs.get("max_attempts") == 7


@pytest.mark.asyncio
async def test_connection_pool_default_max_attempts_is_none() -> None:
    """When ``max_attempts`` is not passed, the pool forwards ``None``
    so ``ClusterClient.connect`` uses its built-in default."""
    pool = ConnectionPool(["a:9001"])
    fake_connect = AsyncMock(return_value=AsyncMock())
    pool._cluster.connect = fake_connect

    await pool._create_connection()

    assert fake_connect.await_args is not None
    call_kwargs = fake_connect.await_args.kwargs
    assert call_kwargs.get("max_attempts") is None


@pytest.mark.asyncio
async def test_create_pool_forwards_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``create_pool(..., max_attempts=N)`` reaches the underlying cluster.connect."""
    captured: dict[str, object] = {}

    async def fake_initialize(self: object) -> None:
        return

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", fake_initialize)
    pool = await create_pool(["a:9001"], max_attempts=5)
    try:

        async def fake_connect(**kwargs: object) -> AsyncMock:
            captured.update(kwargs)
            return AsyncMock()

        pool._cluster.connect = fake_connect  # type: ignore[assignment]
        await pool._create_connection()
        assert captured.get("max_attempts") == 5
    finally:
        await pool.close()


def test_connection_pool_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ConnectionPool(["a:9001"], max_attempts=0)


def test_connection_pool_rejects_negative_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ConnectionPool(["a:9001"], max_attempts=-1)
