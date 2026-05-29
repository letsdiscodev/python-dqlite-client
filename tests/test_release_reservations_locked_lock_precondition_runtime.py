"""Pin: ``_release_reservations_locked`` enforces its lock-held
precondition via a runtime ``raise``, not a bare ``assert`` (stripped
under ``python -O``)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dqliteclient.pool import ConnectionPool


def test_release_reservations_locked_raises_without_lock() -> None:
    """Calling without holding ``_lock`` raises ``AssertionError`` even under ``-O``."""
    pool = ConnectionPool(addresses=["localhost:9001"])
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
    """The source must use ``raise``, not a bare ``assert``, to survive ``python -O``."""
    pool_py = Path(__file__).resolve().parent.parent / "src" / "dqliteclient" / "pool.py"
    source = pool_py.read_text()
    needle = "assert self._lock.locked(), "
    assert needle not in source, (
        "_release_reservations_locked uses a bare ``assert`` for its "
        "lock precondition; this is stripped under ``python -O`` and "
        "the docstring's runtime-enforcement promise is broken. Use "
        "``if not ...: raise AssertionError(...)`` instead."
    )
