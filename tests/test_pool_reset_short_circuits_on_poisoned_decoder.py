"""Pool _reset_connection skips ROLLBACK when the protocol's decoder
is poisoned: is_wire_coherent==False makes _socket_looks_dead True and
_reset_connection returns False without sending ROLLBACK (the next
_read_response would raise ProtocolError anyway — a wasted RTT).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool, _socket_looks_dead


def _make_conn_with_poisoned_protocol(*, in_transaction: bool) -> MagicMock:
    """Stub conn with is_wire_coherent=False but otherwise healthy
    transport, so _socket_looks_dead fires only the wire-coherence branch."""
    transport = MagicMock()
    transport.is_closing.return_value = False
    writer = MagicMock()
    writer.transport = transport
    reader = MagicMock()
    reader.at_eof.return_value = False

    protocol = MagicMock()
    protocol.is_wire_coherent = False  # the load-bearing signal
    protocol._writer = writer
    protocol._reader = reader

    conn = MagicMock()
    conn._protocol = protocol
    conn._address = "localhost:9001"
    conn._in_transaction = in_transaction
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._tx_owner = None
    conn._db_id = 1
    conn.execute = AsyncMock(return_value=None)
    return conn


@pytest.mark.asyncio
async def test_socket_looks_dead_returns_true_on_poisoned_decoder() -> None:
    """_socket_looks_dead returns True on is_wire_coherent=False even
    when transport/reader look healthy."""
    conn = _make_conn_with_poisoned_protocol(in_transaction=True)

    assert _socket_looks_dead(conn) is True


@pytest.mark.asyncio
async def test_reset_connection_skips_rollback_on_poisoned_decoder() -> None:
    """In-tx + poisoned decoder: _reset_connection returns False with no ROLLBACK."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=True)

    result = await pool._reset_connection(conn)

    assert result is False, (
        "poisoned decoder ⇒ ``_reset_connection`` must report the "
        "slot as un-reusable so the pool destroys it"
    )
    assert conn.execute.await_count == 0, (
        "poisoned decoder ⇒ NO wire op — the next _read_response "
        "would just raise ProtocolError, the RTT is wasted"
    )


@pytest.mark.asyncio
async def test_reset_connection_drops_poisoned_decoder_even_without_rollback() -> None:
    """Poisoned wire is dropped even with no open tx — the wire-coherence
    check runs before the needs_rollback calculation."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=False)

    result = await pool._reset_connection(conn)

    assert result is False, (
        "poisoned decoder ⇒ slot is unrecoverable; no point waiting "
        "for the next acquirer to discover the dead wire"
    )
    assert conn.execute.await_count == 0


@pytest.mark.asyncio
async def test_reset_connection_skips_rollback_when_only_savepoint_stack_set() -> None:
    """Short-circuit fires regardless of which flag triggered needs_rollback
    (here _savepoint_stack, not _in_transaction)."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=False)
    conn._savepoint_stack = ["sp1"]

    result = await pool._reset_connection(conn)

    assert result is False
    assert conn.execute.await_count == 0


@pytest.mark.asyncio
async def test_reset_connection_skips_rollback_when_only_untracked_savepoint_flag_set() -> None:
    """Same short-circuit via _has_untracked_savepoint."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=False)
    conn._has_untracked_savepoint = True

    result = await pool._reset_connection(conn)

    assert result is False
    assert conn.execute.await_count == 0


@pytest.mark.asyncio
async def test_reset_connection_attempts_rollback_when_decoder_coherent() -> None:
    """Control: with a coherent decoder the pool DOES issue ROLLBACK."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=True)
    conn._protocol.is_wire_coherent = True

    result = await pool._reset_connection(conn)

    assert result is True, "coherent + clean rollback ⇒ slot is reusable"
    assert conn.execute.await_count == 1
    rollback_call = conn.execute.await_args
    assert "ROLLBACK" in rollback_call.args[0].upper()
