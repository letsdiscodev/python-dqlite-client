"""Direct unit tests for ``retry_with_backoff``'s ``excluded_exceptions``
parameter (otherwise only exercised indirectly via cluster-layer tests)."""

from __future__ import annotations

import pytest

from dqliteclient.retry import retry_with_backoff


class _NonRetryable(ConnectionError):
    """Deterministic-failure subclass of a retryable family."""


class TestExcludedExceptionsShortCircuits:
    async def test_excluded_exception_raises_on_attempt_one(self) -> None:
        """A match re-raises immediately, no sleep or further attempt."""
        calls = 0

        async def raiser() -> None:
            nonlocal calls
            calls += 1
            raise _NonRetryable("no retry")

        with pytest.raises(_NonRetryable):
            await retry_with_backoff(
                raiser,
                retryable_exceptions=(OSError,),
                excluded_exceptions=(_NonRetryable,),
                max_attempts=5,
                base_delay=0.001,
            )
        assert calls == 1

    async def test_excluded_takes_precedence_over_retryable(self) -> None:
        """``excluded_exceptions`` is checked before ``retryable_exceptions``."""
        calls = 0

        async def raiser() -> None:
            nonlocal calls
            calls += 1
            raise _NonRetryable("policy")

        with pytest.raises(_NonRetryable):
            await retry_with_backoff(
                raiser,
                retryable_exceptions=(ConnectionError,),  # parent of _NonRetryable
                excluded_exceptions=(_NonRetryable,),
                max_attempts=5,
                base_delay=0.001,
            )
        assert calls == 1

    async def test_non_excluded_retryable_still_retries(self) -> None:
        """A non-matching exception still retries when exclusions are set."""
        calls = 0

        async def raiser() -> None:
            nonlocal calls
            calls += 1
            raise ConnectionError("transient")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                raiser,
                retryable_exceptions=(ConnectionError,),
                excluded_exceptions=(_NonRetryable,),
                max_attempts=3,
                base_delay=0.001,
            )
        assert calls == 3
