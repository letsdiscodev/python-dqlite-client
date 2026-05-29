"""Input validation for ``retry_with_backoff``: bool / non-finite /
out-of-range values are rejected at the entry point with an actionable
message, not deep inside ``asyncio.sleep``."""

from __future__ import annotations

import pytest

from dqliteclient.retry import retry_with_backoff


async def _ok() -> int:
    return 42


@pytest.mark.asyncio
async def test_max_attempts_must_be_int() -> None:
    with pytest.raises(TypeError, match="max_attempts must be an int"):
        await retry_with_backoff(_ok, max_attempts=1.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_max_attempts_bool_rejected() -> None:
    with pytest.raises(TypeError, match="max_attempts must be an int"):
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
    """The diagnostic must explain WHY 1.0 is rejected."""
    with pytest.raises(ValueError) as exc_info:
        await retry_with_backoff(_ok, jitter=1.0)
    msg = str(exc_info.value)
    assert "exponential-backoff contract" in msg
    assert "zero the backoff" in msg


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
