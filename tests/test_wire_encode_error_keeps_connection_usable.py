"""An encode failure leaves zero bytes on the wire, so the connection stays reusable.

A future streaming encoder would break this invariant.
"""

from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError, DqliteConnectionError
from dqlitewire.exceptions import EncodeError


@pytest.mark.asyncio
async def test_wire_encode_error_does_not_invalidate() -> None:
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 1
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 1))

    async def encode_failing_op(_p: object, _db: int) -> None:
        raise EncodeError("simulated parameter encode failure")

    with pytest.raises(DataError, match="wire encode failed: simulated"):
        await conn._run_protocol(encode_failing_op)

    assert conn._protocol is fake_protocol
    assert conn._invalidation_cause is None


@pytest.mark.asyncio
async def test_wire_encode_error_dataerror_carries_prefix() -> None:
    """The prefix lets an operator distinguish a wire-encoder rejection from
    another caller-side rejection emitting the same string."""
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
    """A transport-level error (not _WireEncodeError) DOES invalidate the connection."""
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
