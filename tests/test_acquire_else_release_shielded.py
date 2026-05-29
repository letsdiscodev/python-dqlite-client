"""Pin: ``ConnectionPool.acquire``'s happy-path ``else`` arm shields
``_release(conn)`` (+ suppresses CancelledError) so a cancel during graceful
exit cannot bypass release and leak the conn / reservation slot."""

from __future__ import annotations

import inspect

from dqliteclient import pool as pool_mod


def test_acquire_else_release_uses_shielded_call() -> None:
    """Source-level pin: ``acquire``'s ``else`` arm calls
    ``asyncio.shield(self._release(conn))``."""
    src = inspect.getsource(pool_mod.ConnectionPool.acquire)
    # Strip comments so an explanatory comment can't satisfy the pin alone.
    code = "\n".join(
        line for line in src.splitlines() if line.strip() and not line.strip().startswith("#")
    )
    assert "shield(self._release(conn))" in code, (
        "acquire's happy-path else arm must wrap _release(conn) in "
        "asyncio.shield (mirroring the exception arm's discipline) so "
        "a cancel landing between the user's __aexit__ return and the "
        "release await does not orphan the conn / reservation slot"
    )
    assert "suppress(asyncio.CancelledError):" in code, (
        "shielded else-arm release must absorb the outer CancelledError "
        "via contextlib.suppress so the inner _release runs to completion"
    )
