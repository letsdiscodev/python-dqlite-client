"""``_run_protocol``'s ``_WireEncodeError`` arm must NOT invalidate
the connection. The wire-layer encoder builds a fully-formed
``bytes`` object before any ``writer.write`` call, so an encode
failure leaves zero bytes on the wire — the connection is reusable.

A future refactor to a streaming encoder would break this invariant.
This test pins the contract; pair with
``test_message_encode_returns_bytes_object`` in the wire suite which
checks the encoder behaviour itself.
"""

from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError, DqliteConnectionError
from dqlitewire.exceptions import EncodeError


@pytest.mark.asyncio
async def test_wire_encode_error_does_not_invalidate() -> None:
    """``_WireEncodeError`` raised inside the ``fn`` callable must
    surface as ``DataError`` and leave the connection usable."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 1
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 1))

    async def encode_failing_op(_p: object, _db: int) -> None:
        # Simulate the wire-layer raising during request construction
        # (e.g. parameter type check) BEFORE any bytes touch the
        # writer.
        raise EncodeError("simulated parameter encode failure")

    with pytest.raises(DataError, match="wire encode failed: simulated"):
        await conn._run_protocol(encode_failing_op)

    # Connection must still be usable: not invalidated, _protocol
    # still set, no invalidation cause.
    assert conn._protocol is fake_protocol
    assert conn._invalidation_cause is None


@pytest.mark.asyncio
async def test_wire_encode_error_dataerror_carries_prefix() -> None:
    """The display surface must carry the ``"wire encode failed: "``
    prefix so an operator triaging a log line can distinguish a
    wire-encoder rejection (bytes never reached the network) from
    another caller-side rejection emitting the same string. Symmetric
    with ``_call_client``'s sibling ``_WireEncodeError`` arm at
    ``dqlitedbapi/cursor.py``."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 1
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 1))

    async def encode_failing_op(_p: object, _db: int) -> None:
        raise EncodeError("encode_int64 expected int, got str")

    with pytest.raises(DataError) as excinfo:
        await conn._run_protocol(encode_failing_op)

    msg = str(excinfo.value)
    assert msg.startswith("wire encode failed: "), (
        f"DataError display must carry the canonical prefix; got {msg!r}"
    )
    assert "encode_int64 expected int, got str" in msg


@pytest.mark.asyncio
async def test_other_protocol_errors_do_invalidate() -> None:
    """Counterpoint: a transport-level error (not _WireEncodeError)
    DOES invalidate the connection. Sanity check the dispatch."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 1
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 1))

    async def transport_failing_op(_p: object, _db: int) -> None:
        raise DqliteConnectionError("simulated transport failure")

    with pytest.raises(DqliteConnectionError):
        await conn._run_protocol(transport_failing_op)

    assert conn._protocol is None
    assert conn._invalidation_cause is not None
