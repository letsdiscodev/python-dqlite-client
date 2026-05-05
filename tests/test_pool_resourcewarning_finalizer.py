"""Pin: ``ConnectionPool`` emits ``ResourceWarning`` when GC'd
without ``await close()`` — but only when at least one slot was
reserved.

Mirrors the dbapi-layer pattern. A pool that was constructed and
never used should NOT emit (false-positive); a pool that warmed
up via initialize() OR reserved a slot via acquire()'s lazy arm
SHOULD emit.
"""

import warnings

from dqliteclient.pool import _pool_unclosed_warning


def test_unclosed_warning_skips_when_never_reserved() -> None:
    closed_flag = [False]
    reserved_flag = [False]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag)
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_skips_when_close_was_called() -> None:
    closed_flag = [True]
    reserved_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag)
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_fires_when_reserved_but_not_closed() -> None:
    closed_flag = [False]
    reserved_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag)
    matching = [r for r in w if issubclass(r.category, ResourceWarning)]
    assert len(matching) == 1
    assert "ConnectionPool" in str(matching[0].message)


def test_finalizer_registered_on_construction() -> None:
    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    assert pool._finalizer is not None
    assert pool._finalizer.alive
    pool._finalizer.detach()
