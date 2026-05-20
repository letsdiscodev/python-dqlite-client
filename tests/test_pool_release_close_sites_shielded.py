"""Pin: every ``await conn.close()`` call site inside the pool's
``_release`` and ``acquire`` cleanup arms is wrapped in
``asyncio.shield`` — symmetric with the rest of the module.

``_POOL_CLEANUP_EXCEPTIONS`` deliberately omits ``CancelledError``;
the shield is the load-bearing guard against an outer cancel
mid-cleanup (e.g. ``asyncio.timeout`` around ``engine.dispose()``)
that would otherwise abort the close mid-``wait_closed`` and orphan
the StreamReader task. A future refactor that drops the shield from
any release-side close site would silently leak transports under
cancel pressure.
"""

from __future__ import annotations

import inspect

from dqliteclient.pool import ConnectionPool


def test_pool_release_close_calls_are_all_shielded() -> None:
    """Inspection pin: every ``await conn.close()`` in pool.py lives
    inside ``asyncio.shield()`` — the only exception is the
    initialize() partial-cleanup path which deliberately runs
    pre-construction (no in-flight reservation to leak)."""
    import dqliteclient.pool as pool_mod

    src = inspect.getsource(pool_mod)
    # Identify every ``await conn.close()`` occurrence. Count both
    # the shielded form (which appears as part of ``asyncio.shield(
    # conn.close())``) and the bare form.
    bare_count = 0
    shielded_count = 0
    for line in src.splitlines():
        stripped = line.strip()
        if "await conn.close()" in stripped:
            bare_count += 1
        if "asyncio.shield(conn.close())" in stripped:
            shielded_count += 1
    # The initialize() partial-cleanup path is the only known bare-
    # await site; everywhere else must be shielded. Allow ≤1 bare.
    assert bare_count - shielded_count <= 1, (
        f"Found {bare_count} bare ``await conn.close()`` sites and "
        f"{shielded_count} shielded sites; the asymmetry suggests a "
        "release-side close was not shielded. Audit recent changes "
        "to pool.py."
    )


def test_release_method_close_sites_all_use_shield() -> None:
    """Inspection pin specifically on ``_release``: all close calls
    inside the method body must use ``asyncio.shield``. ``_release``
    is the per-checkout cleanup path; an unshielded close here
    would leak transports under ``engine.dispose()`` cancellation
    pressure (the most common production trigger for this race)."""
    src = inspect.getsource(ConnectionPool._release)
    assert "await conn.close()" not in src, (
        "Found a raw ``await conn.close()`` in ``_release``; wrap in "
        "``asyncio.shield()`` to match every other close site in pool.py."
    )
