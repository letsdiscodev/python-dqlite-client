"""Pin: ``_invalidate``'s first-cause-wins guard preserves BaseException
causes (CancelledError, KeyboardInterrupt, SystemExit), guarding against
a refactor that tightens the guard to ``isinstance(cause, Exception)``.
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
    """A first BaseException cause survives a subsequent Exception invalidate."""
    conn = DqliteConnection("localhost:9001")
    conn._invalidate(first_cause)
    assert conn._invalidation_cause is first_cause

    follow_up = OperationalError("subsequent transport failure", 1)
    conn._invalidate(follow_up)
    assert conn._invalidation_cause is first_cause


def test_first_exception_preserved_across_baseexception() -> None:
    """Inverse: a first Exception cause survives a subsequent BaseException invalidate."""
    conn = DqliteConnection("localhost:9001")
    first = OSError("first transport error")
    conn._invalidate(first)
    conn._invalidate(asyncio.CancelledError("subsequent cancel"))
    assert conn._invalidation_cause is first
