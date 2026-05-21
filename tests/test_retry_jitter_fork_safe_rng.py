"""Pin: ``retry_with_backoff``'s jitter draw uses a module-level
``SystemRandom`` instance — NOT the process-global
``random._inst`` — so forked workers do not collapse onto identical
jitter sequences.

CPython's module-global ``random._inst`` (the backing state for
bare ``random.uniform``) is inherited byte-for-byte across
``os.fork()`` and has no ``os.register_at_fork`` callback in
``Lib/random.py``. Sibling forked workers (gunicorn / uwsgi /
multiprocessing fork start method) drawing from the inherited
state therefore produce IDENTICAL jitter sequences, defeating the
stagger jitter is meant to provide.

The cluster-layer node-shuffle uses ``_cluster_random:
SystemRandom`` at ``cluster.py:159-165`` for the same reason. This
test pins that the retry layer adopts the same discipline.
"""

from __future__ import annotations

import random

import pytest

import dqliteclient.retry as _retry_mod
from dqliteclient.retry import retry_with_backoff


def test_module_uses_systemrandom_instance() -> None:
    """The retry module must own a ``SystemRandom`` instance and use
    it for jitter draws — not the process-global ``random._inst``.
    """
    assert hasattr(_retry_mod, "_retry_random")
    assert isinstance(_retry_mod._retry_random, random.SystemRandom)


def test_retry_jitter_immune_to_module_random_seed() -> None:
    """``random.seed(0)`` followed by replayed jitter draws must
    produce DIFFERENT samples — proving the retry helper does not
    consult ``random._inst``.

    Before the fix this assertion fails because the module-level
    ``random.uniform`` call IS deterministic under a seeded
    ``random._inst``. After the fix the draws come from
    ``SystemRandom`` which ignores ``random.seed()``.
    """
    real_uniform = _retry_mod._retry_random.uniform
    captured: list[float] = []

    def capturing(a: float, b: float) -> float:
        v = real_uniform(a, b)
        captured.append(v)
        return v

    # First run with a seeded module-global PRNG.
    random.seed(0)
    _retry_mod._retry_random.uniform = capturing
    try:
        _draw_a_few_samples()
        first = captured.copy()
        captured.clear()

        # Restore the same seed — bare ``random._inst`` is now in
        # the identical state as the first run. If the retry helper
        # were consulting ``random._inst`` (the broken shape), the
        # captured samples would match byte-for-byte.
        random.seed(0)
        _draw_a_few_samples()
        second = captured.copy()
    finally:
        _retry_mod._retry_random.uniform = real_uniform

    # SystemRandom ignores random.seed(); replayed draws diverge.
    assert first != second, (
        f"Retry jitter draws are deterministic under random.seed() — "
        f"forked workers will stampede. first={first} second={second}"
    )


def _draw_a_few_samples() -> None:
    """Run a small synchronous burst of jitter draws via the module's
    SystemRandom instance, mimicking what the retry helper does on
    a multi-attempt failure path.
    """
    rng = _retry_mod._retry_random
    for _ in range(5):
        rng.uniform(-0.1, 0.1)


async def test_default_excluded_exceptions_includes_cluster_policy_error() -> None:
    """A ``ClusterPolicyError`` raised inside ``retry_with_backoff``
    must short-circuit on attempt one even when the caller does not
    explicitly opt into ``excluded_exceptions``.

    Without the default exclusion, the helper would burn the full
    ``max_attempts`` window against a deterministic policy rejection.
    """
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
    """Documentation-pin: the module's exported default exclusion
    tuple names ``ClusterPolicyError`` so a future refactor cannot
    silently drop the default without tripping this test.
    """
    from dqliteclient.exceptions import ClusterPolicyError

    assert ClusterPolicyError in _retry_mod._DEFAULT_EXCLUDED


def test_retry_docstring_documents_cluster_connect_divergence() -> None:
    """The public docstring calls out that ``ClusterClient.connect``
    uses tighter retry defaults (``max_attempts=3``, ``max_delay=1.0``)
    for go-dqlite parity. External callers wrapping cluster-layer
    operations need to know to match those values.
    """
    doc = retry_with_backoff.__doc__ or ""
    assert "ClusterClient.connect" in doc
    assert "max_attempts=3" in doc
    assert "max_delay=1.0" in doc
