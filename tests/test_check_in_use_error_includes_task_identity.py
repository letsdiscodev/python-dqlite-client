"""``_check_in_use`` error messages include the offending task reprs."""

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
        assert "another operation is in progress" in msg
        assert "current task: <Task" in msg

    async def test_tx_owner_branch_includes_owner_and_current_task_reprs(
        self,
    ) -> None:
        conn = DqliteConnection("localhost:9001")
        # Distinct sentinel: _check_in_use uses an identity (``is``) comparison.
        owner_sentinel = object()
        conn._in_transaction = True
        conn._tx_owner = owner_sentinel  # type: ignore[assignment]

        with pytest.raises(InterfaceError) as exc_info:
            conn._check_in_use()
        msg = str(exc_info.value)
        assert "owned by another task" in msg
        assert "owner:" in msg
        assert "current:" in msg
        assert repr(owner_sentinel) in msg
