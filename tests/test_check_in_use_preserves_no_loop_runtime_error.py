"""``_check_in_use``'s "no running event loop" arm preserves the
captured ``RuntimeError`` on ``__cause__``.

The arm used to ``raise InterfaceError(...) from None``, suppressing
the underlying asyncio RuntimeError. This dropped a diagnostic-fidelity
signal: future CPython releases could grow a second RuntimeError shape
from ``get_running_loop`` (e.g. ``"loop is closing"``) and the chain
would lose that discriminator. It also created a one-off carve-out
against project discipline (done/ISSUE-207, done/ISSUE-212) that
``from None`` sites must be either documented or upgraded to a
captured cause.

This test pins the upgrade: the InterfaceError's ``__cause__`` is the
original RuntimeError. (``raise X from Y`` always sets
``__suppress_context__`` to True per PEP 3134; the load-bearing
assertion is that ``__cause__`` is no longer ``None``.)
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def test_check_in_use_outside_async_context_preserves_runtime_error_cause() -> None:
    """No running event loop: the InterfaceError carries the captured
    RuntimeError on ``__cause__`` rather than ``None`` (the prior
    ``from None`` behaviour)."""
    conn = DqliteConnection("localhost:9001")
    # Run from a sync context (no running loop) — the guard at the
    # top of ``_check_in_use`` should raise InterfaceError with the
    # captured RuntimeError chained explicitly.
    with pytest.raises(InterfaceError) as excinfo:
        conn._check_in_use()
    assert "must be used from within an async context" in str(excinfo.value)
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError), (
        f"expected RuntimeError on __cause__; got {type(cause).__name__}"
    )
    # The implicit ``__context__`` is the same captured exception —
    # CPython sets it to whatever was active on the except arm.
    assert excinfo.value.__context__ is cause
    # Do not pin the asyncio message text — it is CPython implementation
    # detail. Pinning the class is enough to detect a regression to
    # ``from None`` (which would set __cause__ to None).
