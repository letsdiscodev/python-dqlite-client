"""``_check_in_use``'s no-running-loop arm chains the captured RuntimeError
on ``__cause__`` (it formerly used ``from None``)."""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def test_check_in_use_outside_async_context_preserves_runtime_error_cause() -> None:
    """No running loop: InterfaceError carries the RuntimeError on ``__cause__``."""
    conn = DqliteConnection("localhost:9001")
    with pytest.raises(InterfaceError) as excinfo:
        conn._check_in_use()
    assert "must be used from within an async context" in str(excinfo.value)
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError), (
        f"expected RuntimeError on __cause__; got {type(cause).__name__}"
    )
    assert excinfo.value.__context__ is cause
    # Don't pin the asyncio message text (CPython detail); the class catches the
    # regression to ``from None`` (which sets __cause__ to None).
