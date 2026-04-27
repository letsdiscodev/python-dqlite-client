"""Pool ``_reset_connection`` must skip ROLLBACK when the protocol's
decoder buffer is poisoned.

The pool consults ``protocol.is_wire_coherent`` inside
``_socket_looks_dead`` so a doomed wire is not chased with a wasted
RTT — the next ``_read_response`` would raise ``ProtocolError``
anyway and force an ``_invalidate``. The contract:

* ``is_wire_coherent == False`` →
  ``_socket_looks_dead == True`` →
  ``_reset_connection`` returns False without sending ROLLBACK.

The wire layer pins the decoder's poison/recovery semantics
(``test_server_failure_mid_stream.py``); the pool-side short-circuit
on the poisoned-decoder branch is the only branch in
``_socket_looks_dead`` not directly exercised by an existing test.
A regression that loosens the check (e.g., dropping the
``is_wire_coherent`` line) would silently restore the wasted RTT.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool, _socket_looks_dead


def _make_conn_with_poisoned_protocol(*, in_transaction: bool) -> MagicMock:
    """Build a stub connection whose ``_protocol.is_wire_coherent``
    reports False and whose other transport attributes look healthy
    (so ``_socket_looks_dead`` only fires the wire-coherence branch,
    not a transport-closed branch). Mirrors the real shape closely
    enough for ``_reset_connection`` and ``_socket_looks_dead`` to
    operate."""
    transport = MagicMock()
    transport.is_closing.return_value = False  # transport not closing
    writer = MagicMock()
    writer.transport = transport
    reader = MagicMock()
    reader.at_eof.return_value = False  # reader not at EOF

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
    """Direct unit pin on the helper. ``_socket_looks_dead`` must
    return True when the protocol's wire-coherence accessor reports
    False — even when the transport / reader look healthy."""
    conn = _make_conn_with_poisoned_protocol(in_transaction=True)

    assert _socket_looks_dead(conn) is True


@pytest.mark.asyncio
async def test_reset_connection_skips_rollback_on_poisoned_decoder() -> None:
    """Higher-level pin: with an in-transaction stub conn whose
    decoder is poisoned, ``_reset_connection`` must return False
    WITHOUT sending ROLLBACK over the wire."""
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
async def test_reset_connection_returns_true_when_no_rollback_needed() -> None:
    """Negative pin (control case): when the connection is NOT in a
    transaction and has no savepoint state, ``_reset_connection``
    short-circuits cleanly without consulting the wire-coherence
    accessor — the slot is returned as reusable. Catches a
    regression where the poisoned-decoder branch fires too eagerly."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=False)

    result = await pool._reset_connection(conn)

    assert result is True, "no transaction state ⇒ no ROLLBACK needed ⇒ slot is reusable"
    assert conn.execute.await_count == 0


@pytest.mark.asyncio
async def test_reset_connection_skips_rollback_when_only_savepoint_stack_set() -> None:
    """Pin: the poisoned-decoder short-circuit fires regardless of
    WHICH transaction-tracking flag triggered ``needs_rollback``.
    Use the ``_savepoint_stack`` branch to prove the short-circuit
    isn't gated on ``_in_transaction`` specifically."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=False)
    conn._savepoint_stack = ["sp1"]  # any tracked SAVEPOINT triggers needs_rollback

    result = await pool._reset_connection(conn)

    assert result is False
    assert conn.execute.await_count == 0


@pytest.mark.asyncio
async def test_reset_connection_skips_rollback_when_only_untracked_savepoint_flag_set() -> None:
    """Pin: same coverage extended to ``_has_untracked_savepoint``
    (the quoted-SAVEPOINT autobegun-tx case). The short-circuit must
    drop the slot rather than chase a doomed RTT."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=False)
    conn._has_untracked_savepoint = True

    result = await pool._reset_connection(conn)

    assert result is False
    assert conn.execute.await_count == 0


@pytest.mark.asyncio
async def test_reset_connection_attempts_rollback_when_decoder_coherent() -> None:
    """Negative pin (control): with a coherent decoder, the pool
    DOES issue ROLLBACK over the wire. Confirms the short-circuit
    is the ONLY thing skipping the round-trip — drop it and this
    test stays green, change it the other way and it goes red."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = _make_conn_with_poisoned_protocol(in_transaction=True)
    conn._protocol.is_wire_coherent = True  # flip the signal

    result = await pool._reset_connection(conn)

    assert result is True, "coherent + clean rollback ⇒ slot is reusable"
    assert conn.execute.await_count == 1
    rollback_call = conn.execute.await_args
    assert "ROLLBACK" in rollback_call.args[0].upper()
