"""retry_with_backoff: input validation, exception filtering, context chaining and jitter RNG."""

from __future__ import annotations

import asyncio
import random

import pytest

import dqliteclient.retry as _retry_mod
from dqliteclient.exceptions import (
    ClusterError,
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
)
from dqliteclient.retry import retry_with_backoff

# Tests for retry utilities.


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

        result = await retry_with_backoff(
            fail_then_succeed,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        assert result == "ok"
        assert call_count == 3

    async def test_max_attempts_exceeded(self) -> None:
        call_count = 0

        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await retry_with_backoff(
                always_fail,
                max_attempts=3,
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
            )

        assert call_count == 3

    async def test_max_attempts_one_raises_on_first_failure(self) -> None:
        """With ``max_attempts=1`` the loop breaks on its first iteration."""
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

        assert call_count == 1

    async def test_respects_max_delay(self) -> None:
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
            await original_sleep(0)

        with patch("dqliteclient.retry.asyncio.sleep", side_effect=mock_sleep):
            await retry_with_backoff(
                fail_twice,
                max_attempts=3,
                base_delay=0.1,
                max_delay=0.05,
                jitter=0,
                retryable_exceptions=(ValueError,),
            )

        assert call_count == 3
        assert len(sleep_args) == 2
        # Uncapped delays would be 0.1 and 0.2; both clamp to max_delay=0.05.
        assert sleep_args[0] == pytest.approx(0.05)
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
                retryable_exceptions=(ValueError,),
            )

        assert sleep_args, "expected at least one sleep"
        for d in sleep_args:
            assert d <= 2.0, f"delay {d} exceeded max_delay=2.0"


class TestRetryDefaults:
    """The default retryable set covers transport/cluster errors; bugs propagate."""

    async def test_default_retries_oserror(self) -> None:
        call_count = 0

        async def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("transient")
            return "ok"

        result = await retry_with_backoff(fail_once, base_delay=0.01)
        assert result == "ok"
        assert call_count == 2

    async def test_default_retries_timeout_error_via_oserror(self) -> None:
        """``TimeoutError`` is an ``OSError`` subclass (3.10+), so the
        ``OSError``-only default tuple covers it without a separate entry."""
        call_count = 0

        async def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("connect timed out")
            return "ok"

        result = await retry_with_backoff(fail_once, base_delay=0.01)
        assert result == "ok"
        assert call_count == 2

    async def test_default_does_not_retry_programming_errors(self) -> None:
        import pytest as _pytest

        from dqliteclient.retry import retry_with_backoff

        call_count = 0

        async def buggy() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("bug")

        with _pytest.raises(TypeError, match="bug"):
            await retry_with_backoff(buggy, max_attempts=5, base_delay=0.01)

        assert call_count == 1, (
            "TypeError is outside the default retryable set and must propagate on the first call"
        )


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


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: OperationalError("UNIQUE constraint failed", 1),
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


async def _ok() -> int:
    return 42


@pytest.mark.asyncio
async def test_max_attempts_must_be_int() -> None:
    with pytest.raises(TypeError, match="max_attempts must be int"):
        await retry_with_backoff(_ok, max_attempts=1.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_max_attempts_bool_rejected() -> None:
    with pytest.raises(TypeError, match="max_attempts must be int"):
        await retry_with_backoff(_ok, max_attempts=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-0.1, float("inf"), float("nan")])
async def test_base_delay_bad_values_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="base_delay must be a non-negative finite number"):
        await retry_with_backoff(_ok, base_delay=bad)


@pytest.mark.asyncio
async def test_base_delay_bool_rejected() -> None:
    with pytest.raises(TypeError, match="base_delay must be a number"):
        await retry_with_backoff(_ok, base_delay=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
async def test_max_delay_bad_values_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="max_delay must be a non-negative finite number"):
        await retry_with_backoff(_ok, max_delay=bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.1, float("inf"), float("nan")])
async def test_jitter_bad_values_rejected(bad: float) -> None:
    """Half-open ``[0, 1)``: at ``jitter=1.0`` a draw can zero the backoff."""
    with pytest.raises(ValueError, match=r"jitter must be in \[0, 1\)"):
        await retry_with_backoff(_ok, jitter=bad)


@pytest.mark.asyncio
async def test_jitter_one_message_explains_zero_backoff_risk() -> None:
    with pytest.raises(ValueError) as exc_info:
        await retry_with_backoff(_ok, jitter=1.0)
    assert "jitter must be in [0, 1)" in str(exc_info.value)


@pytest.mark.asyncio
async def test_jitter_zero_point_nine_nine_still_accepted() -> None:
    """The closest practical max-randomisation value is still accepted."""
    result = await retry_with_backoff(_ok, max_attempts=1, jitter=0.99)
    assert result == 42


@pytest.mark.asyncio
async def test_jitter_bool_rejected() -> None:
    with pytest.raises(TypeError, match="jitter must be a number"):
        await retry_with_backoff(_ok, jitter=True)


@pytest.mark.asyncio
async def test_valid_inputs_still_work() -> None:
    result = await retry_with_backoff(_ok, max_attempts=1, base_delay=0.0)
    assert result == 42


@pytest.mark.asyncio
async def test_max_elapsed_seconds_caps_retry_budget() -> None:
    """``max_elapsed_seconds`` aborts the loop once the budget is exceeded."""
    import asyncio
    import time

    calls = 0

    async def _slow_fail() -> int:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        raise OSError("transport refused")

    start = time.monotonic()
    with pytest.raises(OSError):
        await retry_with_backoff(
            _slow_fail,
            max_attempts=100,
            base_delay=0.0,
            max_delay=0.0,
            jitter=0.0,
            max_elapsed_seconds=0.1,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"retry loop blew past its wall-clock budget: {elapsed}s"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -0.1, float("inf"), float("nan")])
async def test_max_elapsed_seconds_bad_values_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="max_elapsed_seconds"):
        await retry_with_backoff(_ok, max_elapsed_seconds=bad)


@pytest.mark.asyncio
async def test_deadline_rechecked_before_each_attempt_after_first() -> None:
    """The deadline is re-checked at the top of each attempt > 0, not only
    before the inter-attempt sleep, so a slow attempt cannot overrun it."""
    import asyncio
    import time

    call_times: list[float] = []
    start = time.monotonic()

    async def _slow_fail() -> int:
        call_times.append(time.monotonic() - start)
        await asyncio.sleep(0.06)
        raise OSError("transport refused")

    with pytest.raises(OSError):
        await retry_with_backoff(
            _slow_fail,
            max_attempts=10,
            base_delay=0.0,
            max_delay=0.0,
            jitter=0.0,
            max_elapsed_seconds=0.05,  # budget < first attempt's 0.06s
        )

    assert len(call_times) == 1, (
        f"Expected exactly one call (deadline reached after first), "
        f"got {len(call_times)} calls at {call_times}"
    )


@pytest.mark.asyncio
async def test_max_elapsed_seconds_bool_rejected() -> None:
    with pytest.raises(TypeError, match="max_elapsed_seconds"):
        await retry_with_backoff(_ok, max_elapsed_seconds=True)


def test_module_uses_systemrandom_instance() -> None:
    """The retry module owns a ``SystemRandom`` instance for jitter draws."""
    assert hasattr(_retry_mod, "_retry_random")
    assert isinstance(_retry_mod._retry_random, random.SystemRandom)


def test_retry_jitter_immune_to_module_random_seed() -> None:
    """Replayed draws after ``random.seed(0)`` differ, proving
    ``SystemRandom`` (not the seeded ``random._inst``) backs jitter."""
    real_uniform = _retry_mod._retry_random.uniform
    captured: list[float] = []

    def capturing(a: float, b: float) -> float:
        v = real_uniform(a, b)
        captured.append(v)
        return v

    random.seed(0)
    _retry_mod._retry_random.uniform = capturing
    try:
        _draw_a_few_samples()
        first = captured.copy()
        captured.clear()

        # Re-seed: if jitter consulted ``random._inst``, draws would match.
        random.seed(0)
        _draw_a_few_samples()
        second = captured.copy()
    finally:
        _retry_mod._retry_random.uniform = real_uniform

    assert first != second, (
        f"Retry jitter draws are deterministic under random.seed() — "
        f"forked workers will stampede. first={first} second={second}"
    )


def _draw_a_few_samples() -> None:
    rng = _retry_mod._retry_random
    for _ in range(5):
        rng.uniform(-0.1, 0.1)


async def test_default_excluded_exceptions_includes_cluster_policy_error() -> None:
    """``ClusterPolicyError`` short-circuits on attempt one via the
    default ``excluded_exceptions``, without the caller opting in."""
    from dqliteclient.exceptions import ClusterPolicyError

    calls = 0

    async def raiser() -> None:
        nonlocal calls
        calls += 1
        raise ClusterPolicyError("blocked")

    with pytest.raises(ClusterPolicyError):
        await retry_with_backoff(
            raiser,
            max_attempts=5,
            base_delay=0.001,
            max_delay=0.01,
        )
    assert calls == 1, (
        f"ClusterPolicyError should short-circuit on attempt one via the "
        f"default ``excluded_exceptions`` tuple, got {calls} calls"
    )


def test_default_excluded_tuple_lists_cluster_policy_error() -> None:
    """The default exclusion tuple names ``ClusterPolicyError``."""
    from dqliteclient.exceptions import ClusterPolicyError

    assert ClusterPolicyError in _retry_mod._DEFAULT_EXCLUDED


async def test_retryable_group_participates_in_retry_loop() -> None:
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
    """A group with any non-retryable leaf must fail-fast, not be masked as transient."""
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
    """A group with any leaf in ``excluded_exceptions`` must fail-fast."""
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
