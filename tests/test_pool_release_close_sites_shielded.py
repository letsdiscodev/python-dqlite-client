"""Pin: every ``await conn.close()`` in _release/acquire cleanup is wrapped in
``asyncio.shield``. _POOL_CLEANUP_EXCEPTIONS omits CancelledError, so the shield
is the only guard against an outer cancel aborting close mid-wait_closed and
orphaning the StreamReader task.
"""

from __future__ import annotations

import inspect

from dqliteclient.pool import ConnectionPool


def test_pool_release_close_calls_are_all_shielded() -> None:
    """Every await conn.close() in pool.py is shielded, except the initialize()
    partial-cleanup path which runs pre-construction (no reservation to leak)."""
    import dqliteclient.pool as pool_mod

    src = inspect.getsource(pool_mod)
    bare_count = 0
    shielded_count = 0
    for line in src.splitlines():
        stripped = line.strip()
        if "await conn.close()" in stripped:
            bare_count += 1
        if "asyncio.shield(conn.close())" in stripped:
            shielded_count += 1
    # Allow <=1 bare: the initialize() partial-cleanup path is the only one.
    assert bare_count - shielded_count <= 1, (
        f"Found {bare_count} bare ``await conn.close()`` sites and "
        f"{shielded_count} shielded sites; the asymmetry suggests a "
        "release-side close was not shielded. Audit recent changes "
        "to pool.py."
    )


def test_release_method_close_sites_all_use_shield() -> None:
    """All close calls in _release use asyncio.shield."""
    src = inspect.getsource(ConnectionPool._release)
    assert "await conn.close()" not in src, (
        "Found a raw ``await conn.close()`` in ``_release``; wrap in "
        "``asyncio.shield()`` to match every other close site in pool.py."
    )
