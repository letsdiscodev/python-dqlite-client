"""``retry_with_backoff``'s exhaustion path chains prior-attempt
failures via ``BaseExceptionGroup`` so the diagnostic for an
exhausted retry loop carries every attempt's error, not just the
last one. Mirrors the per-node aggregate discipline in
``_find_leader_impl`` and ``ConnectionPool.initialize``.
"""

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.retry import retry_with_backoff


@pytest.mark.asyncio
async def test_retry_exhaustion_chains_via_bounded_group() -> None:
    """The exhaustion raise's ``__cause__`` is a
    ``BaseExceptionGroup`` carrying every attempt when ``len > 1``."""
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
    # Each child carries the per-attempt message.
    msgs = [str(e) for e in cause.exceptions]
    for i in range(1, 4):
        assert any(f"attempt {i}" in m for m in msgs)


@pytest.mark.asyncio
async def test_single_attempt_path_no_group_wrap() -> None:
    """When max_attempts=1, the single-attempt failure raises
    directly without a chain group — preserves the simple-case shape
    callers are used to."""

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

    # No chain group wrap on single attempt.
    assert excinfo.value.__cause__ is None
