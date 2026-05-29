"""Pin: jitter draws use a module-level ``SystemRandom``, not the
process-global ``random._inst`` (inherited byte-for-byte across
``os.fork()``, which would make forked workers share jitter sequences)."""

from __future__ import annotations

import random

import pytest

import dqliteclient.retry as _retry_mod
from dqliteclient.retry import retry_with_backoff


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
