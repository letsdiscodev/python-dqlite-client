"""Input validation for ``retry_with_backoff``. Rejects bool /
non-finite / out-of-range values at the entry point rather than
letting them propagate into ``asyncio.sleep``, where the error
surface is ``ValueError: sleep length must be non-negative`` with
no hint about the caller's misuse.
"""

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
        await retry_with_backoff(_ok, max_attempts=True)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-0.1, float("inf"), float("nan")])
async def test_base_delay_bad_values_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="base_delay must be a non-negative finite number"):
        await retry_with_backoff(_ok, base_delay=bad)


@pytest.mark.asyncio
async def test_base_delay_bool_rejected() -> None:
    with pytest.raises(TypeError, match="base_delay must be a number"):
        await retry_with_backoff(_ok, base_delay=True)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
async def test_max_delay_bad_values_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="max_delay must be a non-negative finite number"):
        await retry_with_backoff(_ok, max_delay=bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-0.1, 1.1, float("inf"), float("nan")])
async def test_jitter_bad_values_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"jitter must be in \[0, 1\]"):
        await retry_with_backoff(_ok, jitter=bad)


@pytest.mark.asyncio
async def test_jitter_bool_rejected() -> None:
    with pytest.raises(TypeError, match="jitter must be a number"):
        await retry_with_backoff(_ok, jitter=True)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_valid_inputs_still_work() -> None:
    result = await retry_with_backoff(_ok, max_attempts=1, base_delay=0.0)
    assert result == 42
