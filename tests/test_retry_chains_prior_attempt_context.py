"""``retry_with_backoff``'s exhaustion path chains every attempt's
failure via ``BaseExceptionGroup``, not just the last one."""

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.retry import retry_with_backoff


@pytest.mark.asyncio
async def test_retry_exhaustion_chains_via_bounded_group() -> None:
    """``__cause__`` is a ``BaseExceptionGroup`` carrying every attempt when ``len > 1``."""
    attempts = 0

    async def always_fail() -> None:
        nonlocal attempts
        attempts += 1
        raise DqliteConnectionError(f"attempt {attempts}")

    with pytest.raises(DqliteConnectionError) as excinfo:
        await retry_with_backoff(
            always_fail,
            max_attempts=3,
            retryable_exceptions=(DqliteConnectionError,),
            base_delay=0.0,
            max_delay=0.0,
            jitter=0.0,
        )

    cause = excinfo.value.__cause__
    assert isinstance(cause, BaseExceptionGroup)
    assert len(cause.exceptions) == 3
    msgs = [str(e) for e in cause.exceptions]
    for i in range(1, 4):
        assert any(f"attempt {i}" in m for m in msgs)


@pytest.mark.asyncio
async def test_single_attempt_path_no_group_wrap() -> None:
    """With max_attempts=1 the failure raises directly, no chain group."""

    async def fail_once() -> None:
        raise DqliteConnectionError("only attempt")

    with pytest.raises(DqliteConnectionError, match="only attempt") as excinfo:
        await retry_with_backoff(
            fail_once,
            max_attempts=1,
            retryable_exceptions=(DqliteConnectionError,),
            base_delay=0.0,
            max_delay=0.0,
            jitter=0.0,
        )

    assert excinfo.value.__cause__ is None
