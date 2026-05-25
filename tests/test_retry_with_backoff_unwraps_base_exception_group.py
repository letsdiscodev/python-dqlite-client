"""Pin: ``retry_with_backoff`` unwraps ``BaseExceptionGroup`` via
``split(retryable_exceptions)`` so a structured-concurrency wrapped
operation (``asyncio.TaskGroup``) whose every leaf is retryable
participates in the retry loop.

Pre-fix the bare ``except retryable_exceptions as e:`` arm matched
by ``isinstance`` only — ``BaseExceptionGroup`` is not a subclass
of the leaf classes, so the group propagated past the retry envelope
on the first attempt. PEP 654 / Python 3.11+ added the group type
and the ``except*`` syntax; this fix uses the ``split`` API to
participate in the discipline without depending on ``except*``.

Mirrors the recursive-unwrap pattern at ``_run_protocol``'s
cancel-detection arm.
"""

from __future__ import annotations

import pytest

from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqliteclient.retry import retry_with_backoff

pytestmark = pytest.mark.asyncio


async def test_retryable_group_participates_in_retry_loop() -> None:
    """A ``BaseExceptionGroup`` whose every leaf is in
    ``retryable_exceptions`` is unwrapped and the retry loop runs.
    """
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise BaseExceptionGroup(
                "tg failures",
                [
                    DqliteConnectionError("transient 1"),
                    DqliteConnectionError("transient 2"),
                ],
            )
        return "ok"

    result = await retry_with_backoff(
        flaky,
        retryable_exceptions=(DqliteConnectionError,),
        max_attempts=5,
        base_delay=0.001,
        jitter=0,
    )
    assert result == "ok"
    assert attempts == 3


async def test_mixed_group_with_non_retryable_leaf_fails_fast() -> None:
    """A group containing any non-retryable leaf must propagate on
    the first attempt — silently retrying would mask the
    deterministic failure as transient.
    """
    attempts = 0

    async def mixed_failure() -> str:
        nonlocal attempts
        attempts += 1
        raise BaseExceptionGroup(
            "mixed",
            [
                DqliteConnectionError("transient"),
                ValueError("deterministic"),
            ],
        )

    with pytest.raises(BaseExceptionGroup):
        await retry_with_backoff(
            mixed_failure,
            retryable_exceptions=(DqliteConnectionError,),
            max_attempts=3,
            base_delay=0.001,
            jitter=0,
        )
    assert attempts == 1, f"mixed group must fail-fast on first attempt; got {attempts} attempts"


async def test_group_with_any_excluded_leaf_fails_fast() -> None:
    """A group with any leaf in ``excluded_exceptions`` (subclass of
    retryable but explicitly opt-out) must propagate on first attempt.
    """
    attempts = 0

    class _NonTransient(DqliteConnectionError):
        pass

    async def deterministic_group() -> str:
        nonlocal attempts
        attempts += 1
        raise BaseExceptionGroup(
            "with excluded",
            [
                _NonTransient("deterministic"),
                DqliteConnectionError("transient"),
            ],
        )

    with pytest.raises(BaseExceptionGroup):
        await retry_with_backoff(
            deterministic_group,
            retryable_exceptions=(DqliteConnectionError,),
            excluded_exceptions=(_NonTransient,),
            max_attempts=3,
            base_delay=0.001,
            jitter=0,
        )
    assert attempts == 1, (
        f"group with excluded leaf must fail-fast on first attempt; got {attempts}"
    )


async def test_leaf_retryable_path_still_works() -> None:
    """The legacy non-group leaf retryable path is unchanged."""
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise OperationalError("transient", 1)
        return "ok"

    result = await retry_with_backoff(
        flaky,
        retryable_exceptions=(OperationalError,),
        max_attempts=3,
        base_delay=0.001,
        jitter=0,
    )
    assert result == "ok"
    assert attempts == 2
