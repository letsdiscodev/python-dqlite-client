"""Pin: ``_release_reservations_locked`` enforces the lock-held
precondition at runtime, not via a bare ``assert`` (which is stripped
under ``python -O``).

The function's own docstring promises immediate raise on
lock-not-held; that contract must hold under any Python invocation,
including optimised mode. A future maintainer reverting to
``assert self._lock.locked(), ...`` would silently re-introduce the
strip-under-`-O` hole.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dqliteclient.pool import ConnectionPool


def test_release_reservations_locked_raises_without_lock() -> None:
    """Calling the helper without holding ``_lock`` raises
    ``AssertionError`` even under ``-O``. The check is a runtime
    ``raise``, not a bare ``assert``."""
    pool = ConnectionPool(addresses=["localhost:9001"])
    # The lock starts unlocked; calling the helper directly should
    # surface the precondition violation.
    expected = "_release_reservations_locked called without _lock held"
    with pytest.raises(AssertionError, match=expected):
        pool._release_reservations_locked(1)


def test_release_reservations_locked_succeeds_with_lock() -> None:
    """Positive control: the helper succeeds when the lock is held."""

    async def run() -> None:
        pool = ConnectionPool(addresses=["localhost:9001"])
        async with pool._lock:
            pool._size = 1
            pool._release_reservations_locked(1)
            assert pool._size == 0

    asyncio.run(run())


def test_pool_source_uses_runtime_check_not_bare_assert() -> None:
    """Static-discipline pin: the source must use ``raise AssertionError(...)``
    or ``if not ...: raise``, NOT a bare ``assert self._lock.locked()``,
    so the precondition survives ``python -O`` stripping."""
    pool_py = Path(__file__).resolve().parent.parent / "src" / "dqliteclient" / "pool.py"
    source = pool_py.read_text()
    needle = "assert self._lock.locked(), "
    assert needle not in source, (
        "_release_reservations_locked uses a bare ``assert`` for its "
        "lock precondition; this is stripped under ``python -O`` and "
        "the docstring's runtime-enforcement promise is broken. Use "
        "``if not ...: raise AssertionError(...)`` instead."
    )
