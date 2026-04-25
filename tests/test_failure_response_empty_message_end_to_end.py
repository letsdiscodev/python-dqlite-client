"""End-to-end pin: ``FailureResponse`` with empty/whitespace message
renders the placeholder text through every protocol path.

The wire codec round-trips empty messages without issue, and the
unit-level ``_failure_message`` helper is pinned in
``test_protocol_failure_messages.py``. The integration through every
protocol dispatcher (exec_sql / query_sql / etc.) is not pinned end-
to-end. Add the missing pins so a regression in the helper invocation
surfaces immediately.
"""

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
    p._handshake_done = True
    return p


@pytest.mark.parametrize("code", [0, 1, 19, 10250])
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


@pytest.mark.parametrize("code", [0, 1, 19, 10250])
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


async def test_finalize_empty_message_renders_placeholder(protocol: DqliteProtocol) -> None:
    protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
        code=21, message=""
    ).encode()
    with pytest.raises(OperationalError) as exc_info:
        await protocol.finalize(1, 7)
    assert "(no diagnostic from server)" in str(exc_info.value)
    assert exc_info.value.code == 21


async def test_prepare_empty_message_renders_placeholder(protocol: DqliteProtocol) -> None:
    protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
        code=1, message=""
    ).encode()
    with pytest.raises(OperationalError) as exc_info:
        await protocol.prepare(1, "SELECT 1")
    assert "(no diagnostic from server)" in str(exc_info.value)
    assert exc_info.value.code == 1
