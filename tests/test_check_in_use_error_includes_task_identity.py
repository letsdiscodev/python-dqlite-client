"""``_check_in_use`` error messages include the offending task names.

When two coroutines accidentally share a connection, the historical
diagnostic surface ("another operation is in progress" / "owned by
another task") was correct but lacked the actual task identities.
Operators triaging a concurrent-access bug had to walk the call graph
to figure out which two tasks were involved.

Pin the richer diagnostic: the message now includes the
``repr(asyncio.current_task())`` of the offending caller, and (for the
tx-owner branch) the ``repr(self._tx_owner)`` of the holder. The
existing tx-009 substring contracts ("another operation is in
progress", "owned by another task") are preserved verbatim so any
downstream code that branches on the wording still works.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
class TestCheckInUseIncludesTaskIdentity:
    async def test_in_use_branch_includes_current_task_repr(self) -> None:
        conn = DqliteConnection("localhost:9001")
        conn._in_use = True
        with pytest.raises(InterfaceError) as exc_info:
            conn._check_in_use()
        msg = str(exc_info.value)
        # Existing tx-009 substring contract preserved.
        assert "another operation is in progress" in msg
        # New: the current task's repr appears so operators can
        # identify the offending caller without walking the call
        # graph. Inside an asyncio test, current_task() is non-None
        # and its repr starts with "<Task".
        assert "current task: <Task" in msg

    async def test_tx_owner_branch_includes_owner_and_current_task_reprs(
        self,
    ) -> None:
        conn = DqliteConnection("localhost:9001")
        # Build a deliberately distinct owner sentinel; identity ``is``
        # comparison in _check_in_use distinguishes it from the
        # current task.
        owner_sentinel = object()
        conn._in_transaction = True
        conn._tx_owner = owner_sentinel  # type: ignore[assignment]

        with pytest.raises(InterfaceError) as exc_info:
            conn._check_in_use()
        msg = str(exc_info.value)
        # Existing tx-009 substring contract preserved.
        assert "owned by another task" in msg
        # Both the owner repr and the current task repr appear so
        # operators can correlate which two tasks raced.
        assert "owner:" in msg
        assert "current:" in msg
        assert repr(owner_sentinel) in msg
