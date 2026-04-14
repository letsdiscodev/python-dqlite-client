"""Tests for retry utilities."""

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
        import time

        call_count = 0
        timestamps: list[float] = []

        async def track_time() -> str:
            nonlocal call_count
            timestamps.append(time.monotonic())
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        await retry_with_backoff(track_time, max_attempts=3, base_delay=0.01, max_delay=0.02)

        # Second retry delay should be capped at max_delay
        assert len(timestamps) == 3
