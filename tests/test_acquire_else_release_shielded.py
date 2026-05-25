"""Pin: ``ConnectionPool.acquire``'s happy-path ``else`` arm wraps
``await self._release(conn)`` in ``asyncio.shield`` +
``contextlib.suppress(asyncio.CancelledError)``, mirroring the
exception-arm shielding at ``pool.py:2265-2271``.

Pre-fix, a cancel landing between the user's ``__aexit__`` return and
``_release(conn)``'s body raised at the unshielded await, bypassing
``_release`` entirely. The conn stayed checked out (``_pool_released
= False``), the reservation slot stayed held, and the transport was
never returned to the queue nor closed — a one-conn leak per
cancel-during-graceful-exit event. Under sustained
``asyncio.timeout()`` around ``async with pool.acquire()`` patterns
the pool's ``_size`` drifts upward and ``max_size`` becomes a soft
cap.
"""

from __future__ import annotations

import inspect

from dqliteclient import pool as pool_mod


def test_acquire_else_release_uses_shielded_call() -> None:
    """Source-level pin: ``acquire``'s ``else`` arm calls
    ``asyncio.shield(self._release(conn))`` (not the bare
    ``await self._release(conn)``).

    The exception arm already uses the shielded shape — this pin
    closes the asymmetry on the happy-path arm.
    """
    src = inspect.getsource(pool_mod.ConnectionPool.acquire)
    # Locate the "else:" block following the "except BaseException:".
    # The fix wraps the release inside the else with shield.
    # Strip comments so an explanatory comment doesn't satisfy the pin
    # without the code being present.
    code = "\n".join(
        line for line in src.splitlines() if line.strip() and not line.strip().startswith("#")
    )
    assert "shield(self._release(conn))" in code, (
        "acquire's happy-path else arm must wrap _release(conn) in "
        "asyncio.shield (mirroring the exception arm's discipline) so "
        "a cancel landing between the user's __aexit__ return and the "
        "release await does not orphan the conn / reservation slot"
    )
    # The CancelledError suppression must accompany the shield — the
    # shield protects the inner task; the suppress absorbs the
    # CancelledError raised at the outer await.
    assert "suppress(asyncio.CancelledError):" in code, (
        "shielded else-arm release must absorb the outer CancelledError "
        "via contextlib.suppress so the inner _release runs to completion"
    )
