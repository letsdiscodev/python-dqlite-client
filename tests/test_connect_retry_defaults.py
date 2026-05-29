"""``ClusterClient.connect`` retry backoff is capped at 1 s with an optional
``max_elapsed_seconds`` wall-clock cap; ``max_attempts=None`` keeps the default-3
semantic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient import create_pool
from dqliteclient.cluster import _DEFAULT_CONNECT_MAX_DELAY, ClusterClient
from dqliteclient.exceptions import DqliteConnectionError


def test_default_connect_max_delay_is_1_second() -> None:
    assert _DEFAULT_CONNECT_MAX_DELAY == 1.0


@pytest.mark.asyncio
async def test_connect_caps_per_iteration_backoff_at_1s() -> None:
    """With ``max_attempts=8`` every scheduled backoff sleep must be <= 1 s."""
    cluster = ClusterClient.from_addresses(["localhost:9001"], timeout=0.5)

    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def fail_find_leader(**_kw: object) -> str:
        raise OSError("connection refused")

    cluster.find_leader = AsyncMock(side_effect=fail_find_leader)

    with (
        patch("dqliteclient.retry.asyncio.sleep", side_effect=record_sleep),
        pytest.raises(OSError),
    ):
        await cluster.connect(max_attempts=8)

    # Small epsilon for jitter applied after the cap clamp.
    assert sleeps, "expected at least one backoff sleep"
    for s in sleeps:
        assert s <= _DEFAULT_CONNECT_MAX_DELAY * 1.05, (
            f"backoff sleep {s:.3f}s exceeds 1s cap (with 5% jitter margin)"
        )


@pytest.mark.asyncio
async def test_max_attempts_none_unchanged_uses_default_three() -> None:
    """``max_attempts=None`` selects the default of 3, NOT unbounded."""
    cluster = ClusterClient.from_addresses(["localhost:9001"], timeout=0.5)

    call_count = 0

    async def count_and_fail(**_kw: object) -> str:
        nonlocal call_count
        call_count += 1
        raise OSError("connection refused")

    cluster.find_leader = AsyncMock(side_effect=count_and_fail)

    with (
        patch("dqliteclient.retry.asyncio.sleep", new=AsyncMock(return_value=None)),
        pytest.raises(OSError),
    ):
        await cluster.connect(max_attempts=None)

    assert call_count == 3, f"expected exactly 3 attempts; got {call_count}"


@pytest.mark.asyncio
async def test_connect_max_elapsed_seconds_caps_total_wall_clock() -> None:
    """With ``max_attempts=100`` and ``max_elapsed_seconds=0.3`` the loop must
    abort on the elapsed budget, not run 100 attempts."""
    cluster = ClusterClient.from_addresses(["localhost:9001"], timeout=0.5)

    async def fail_find_leader(**_kw: object) -> str:
        raise OSError("connection refused")

    cluster.find_leader = AsyncMock(side_effect=fail_find_leader)

    start = asyncio.get_running_loop().time()
    with pytest.raises((OSError, DqliteConnectionError)):
        await cluster.connect(max_attempts=100, max_elapsed_seconds=0.3)
    elapsed = asyncio.get_running_loop().time() - start

    # 2 s slack absorbs jitter; the regression vector is 100 x 1 s = 100 s.
    assert elapsed < 2.0, f"expected elapsed < 2s; got {elapsed:.3f}s"


@pytest.mark.parametrize(
    "bad_value",
    [
        True,  # bool slips through isinstance(_, (int, float))
        False,
        0,
        -1.0,
        float("inf"),
        float("nan"),
    ],
)
def test_max_elapsed_seconds_validation_rejects_bad_values(bad_value: object) -> None:
    """Reject bool, zero, negative, and non-finite values at the
    public ``connect()`` call site."""
    cluster = ClusterClient.from_addresses(["localhost:9001"])

    async def run() -> None:
        with pytest.raises((TypeError, ValueError), match="max_elapsed_seconds"):
            await cluster.connect(max_elapsed_seconds=bad_value)  # type: ignore[arg-type]

    asyncio.run(run())


def test_max_elapsed_seconds_bool_rejected_with_typeerror() -> None:
    cluster = ClusterClient.from_addresses(["localhost:9001"])

    async def run() -> None:
        with pytest.raises(TypeError, match="max_elapsed_seconds"):
            await cluster.connect(max_elapsed_seconds=True)

    asyncio.run(run())


@pytest.mark.asyncio
async def test_create_pool_forwards_max_elapsed_seconds() -> None:
    """``create_pool`` forwards ``max_elapsed_seconds`` to the pool."""
    pool = await create_pool(
        addresses=["localhost:9001"],
        max_elapsed_seconds=1.5,
        min_size=0,
        max_size=1,
    )
    try:
        assert pool._max_elapsed_seconds == 1.5
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_max_elapsed_seconds_validation() -> None:
    """The pool validates at init, not at first acquire."""
    with pytest.raises(TypeError, match="max_elapsed_seconds"):
        await create_pool(
            addresses=["localhost:9001"],
            max_elapsed_seconds=True,
            min_size=0,
            max_size=1,
        )

    with pytest.raises(ValueError, match="max_elapsed_seconds"):
        await create_pool(
            addresses=["localhost:9001"],
            max_elapsed_seconds=-1.0,
            min_size=0,
            max_size=1,
        )


@pytest.mark.asyncio
async def test_pool_default_max_elapsed_seconds_is_none() -> None:
    pool = await create_pool(addresses=["localhost:9001"], min_size=0, max_size=1)
    try:
        assert pool._max_elapsed_seconds is None
    finally:
        await pool.close()
