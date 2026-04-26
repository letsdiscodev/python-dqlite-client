"""The default retry set must exclude deterministic server / client
errors (``OperationalError``, ``DataError``, ``InterfaceError``).

Retrying a constraint violation or a type mismatch burns the full
exponential-backoff budget before the caller sees the underlying
cause. Only transport- and cluster-level errors belong in the
default tuple; callers that really want broader catches opt in
explicitly.
"""

from __future__ import annotations

import pytest

from dqliteclient.exceptions import (
    ClusterError,
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
)
from dqliteclient.retry import retry_with_backoff


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: OperationalError(1, "UNIQUE constraint failed"),
        lambda: DataError("int64 out of range"),
        lambda: InterfaceError("concurrent access"),
    ],
)
@pytest.mark.asyncio
async def test_default_does_not_retry_deterministic_errors(exc_factory) -> None:
    call_count = 0

    async def always_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise exc_factory()

    exc = exc_factory()
    with pytest.raises(type(exc)):
        await retry_with_backoff(always_fail, max_attempts=5, base_delay=0.001)
    assert call_count == 1, (
        "deterministic server/client errors must not be retried by the "
        f"default tuple; saw {call_count} attempts"
    )


@pytest.mark.asyncio
async def test_default_retries_dqlite_connection_error() -> None:
    call_count = 0

    async def always_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise DqliteConnectionError("transport down")

    with pytest.raises(DqliteConnectionError):
        await retry_with_backoff(always_fail, max_attempts=3, base_delay=0.001)
    assert call_count == 3


@pytest.mark.asyncio
async def test_default_retries_cluster_error() -> None:
    call_count = 0

    async def always_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise ClusterError("no leader yet")

    with pytest.raises(ClusterError):
        await retry_with_backoff(always_fail, max_attempts=3, base_delay=0.001)
    assert call_count == 3
