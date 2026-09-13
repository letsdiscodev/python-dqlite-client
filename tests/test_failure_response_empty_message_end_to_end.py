"""``FailureResponse`` with an empty/whitespace message renders the
placeholder text through every protocol path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    p = DqliteProtocol(reader, writer)
    return p


@pytest.mark.parametrize("code", [1, 19, 10250])
@pytest.mark.parametrize("message", ["", "   ", "\t\n"])
async def test_exec_sql_empty_message_renders_placeholder(
    protocol: DqliteProtocol, code: int, message: str
) -> None:
    protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
        code=code, message=message
    ).encode()
    with pytest.raises(OperationalError) as exc_info:
        await protocol.exec_sql(1, "SELECT 1")
    assert "(no diagnostic from server)" in str(exc_info.value)
    assert exc_info.value.code == code


@pytest.mark.parametrize("code", [1, 19, 10250])
async def test_query_sql_empty_message_renders_placeholder(
    protocol: DqliteProtocol, code: int
) -> None:
    protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
        code=code, message=""
    ).encode()
    with pytest.raises(OperationalError) as exc_info:
        await protocol.query_sql(1, "SELECT 1")
    assert "(no diagnostic from server)" in str(exc_info.value)
    assert exc_info.value.code == code
