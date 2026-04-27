"""Pin: ``_invalidate(cause)``'s first-cause-wins guard preserves
``BaseException``-not-``Exception`` causes (CancelledError,
KeyboardInterrupt, SystemExit) across a subsequent invalidate from
an Exception subclass.

The contract docstring at ``connection.py`` describes the cancel-
during-COMMIT case where the first invalidation's cause is
``CancelledError``. A future refactor that tightened the guard to
``isinstance(cause, Exception)`` would silently drop the cancel /
interrupt context — there's no test today that catches that.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError


@pytest.mark.parametrize(
    "first_cause",
    [
        asyncio.CancelledError("cancel"),
        KeyboardInterrupt("ctrl-c"),
        SystemExit(1),
    ],
)
def test_first_baseexception_cause_preserved_across_subsequent_exception(
    first_cause: BaseException,
) -> None:
    """A first-invalidation cause from a BaseException-not-Exception
    subclass MUST survive a subsequent invalidate from an Exception
    subclass — the guard at ``connection.py`` is type-agnostic on
    ``cause is not None and self._invalidation_cause is None``."""
    conn = DqliteConnection("localhost:9001")
    conn._invalidate(first_cause)
    assert conn._invalidation_cause is first_cause

    follow_up = OperationalError(1, "subsequent transport failure")
    conn._invalidate(follow_up)
    assert conn._invalidation_cause is first_cause


def test_first_exception_preserved_across_baseexception() -> None:
    """Inverse: a first ordinary-Exception cause is preserved when a
    subsequent BaseException invalidation lands."""
    conn = DqliteConnection("localhost:9001")
    first = OSError("first transport error")
    conn._invalidate(first)
    conn._invalidate(asyncio.CancelledError("subsequent cancel"))
    assert conn._invalidation_cause is first
