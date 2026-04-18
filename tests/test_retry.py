"""Tests for retry utilities."""

import asyncio

import pytest

from dqliteclient.retry import retry_with_backoff


class TestRetryWithBackoff:
    async def test_success_first_try(self) -> None:
        call_count = 0

        async def success() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(success)
        assert result == "ok"
        assert call_count == 1

    async def test_success_after_retries(self) -> None:
        call_count = 0

        async def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        result = await retry_with_backoff(fail_then_succeed, base_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    async def test_max_attempts_exceeded(self) -> None:
        call_count = 0

        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await retry_with_backoff(always_fail, max_attempts=3, base_delay=0.01)

        assert call_count == 3

    async def test_max_attempts_one_raises_on_first_failure(self) -> None:
        """Edge case: with ``max_attempts=1`` the loop breaks on its first
        iteration. Covers the ``if attempt == max_attempts - 1: break``
        path that the final ``raise last_error`` relies on.
        """
        call_count = 0

        async def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await retry_with_backoff(fail_once, max_attempts=1, base_delay=0.01)

        assert call_count == 1

    async def test_max_attempts_zero_raises_value_error(self) -> None:
        async def should_not_be_called() -> str:
            raise AssertionError("Should not be called with max_attempts=0")

        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            await retry_with_backoff(should_not_be_called, max_attempts=0)

    async def test_non_retryable_exception_fails_immediately(self) -> None:
        call_count = 0

        async def raise_type_error() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("bug")

        with pytest.raises(TypeError, match="bug"):
            await retry_with_backoff(
                raise_type_error,
                max_attempts=5,
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
            )

        assert call_count == 1  # Should not retry

    async def test_respects_max_delay(self) -> None:
        """Verify that the backoff delay is capped at max_delay."""
        from unittest.mock import patch

        call_count = 0
        sleep_args: list[float] = []

        async def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_args.append(delay)
            await original_sleep(0)  # Yield control without actual delay

        with patch("dqliteclient.retry.asyncio.sleep", side_effect=mock_sleep):
            await retry_with_backoff(
                fail_twice, max_attempts=3, base_delay=0.1, max_delay=0.05, jitter=0
            )

        assert call_count == 3
        assert len(sleep_args) == 2  # Two retries = two sleeps
        # First delay: 0.1 * 2^0 = 0.1, capped to 0.05
        assert sleep_args[0] == pytest.approx(0.05)
        # Second delay: 0.1 * 2^1 = 0.2, capped to 0.05
        assert sleep_args[1] == pytest.approx(0.05)

    async def test_jitter_does_not_exceed_max_delay(self) -> None:
        """max_delay is a hard ceiling: even with jitter, realized delay must not exceed it."""
        from unittest.mock import patch

        sleep_args: list[float] = []

        async def always_fail() -> str:
            raise ValueError("fail")

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_args.append(delay)
            await original_sleep(0)

        # Force jitter to its positive endpoint so the multiplier is (1 + jitter)
        def max_jitter(_low: float, high: float) -> float:
            return high

        with (
            patch("dqliteclient.retry.asyncio.sleep", side_effect=mock_sleep),
            patch("dqliteclient.retry.random.uniform", side_effect=max_jitter),
            pytest.raises(ValueError, match="fail"),
        ):
            await retry_with_backoff(
                always_fail,
                max_attempts=10,
                base_delay=1.0,
                max_delay=2.0,
                jitter=0.1,
            )

        # Several attempts will hit the cap; none should exceed max_delay.
        assert sleep_args, "expected at least one sleep"
        for d in sleep_args:
            assert d <= 2.0, f"delay {d} exceeded max_delay=2.0"
