"""``_read_continuation``'s server-FAILURE arm must call ``_failure_text``
(pre-truncate body before appending the addr suffix) so the peer-address
suffix survives ``OperationalError``'s 1024-codepoint display cap even when
the server's ``FailureResponse.message`` is near the 64 KiB wire cap."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.exceptions import OperationalError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import RowsResponse


def _make_proto_with_address(address: str) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = address
    return proto


@pytest.mark.asyncio
async def test_continuation_failure_arm_preserves_addr_suffix_at_wire_cap() -> None:
    """A near-wire-cap FAILURE mid-continuation: OperationalError.message must
    still end with the peer-address suffix."""
    from dqlitewire.exceptions import ServerFailure as _WireServerFailure

    proto = _make_proto_with_address("host-c:19003")

    # 65000-char message (> 1024 display cap) so without pre-truncation the
    # suffix is pushed off the display.
    class _StubDecoder:
        def decode_continuation(
            self,
        ) -> RowsResponse | _WireServerFailure | None:
            raise _WireServerFailure(code=1001, message="X" * 65000)

        def feed(self, data: bytes) -> None:
            pass

    proto._decoder = _StubDecoder()  # type: ignore[assignment]

    async def _stub_read_data(deadline: float) -> bytes:
        return b""

    proto._read_data = _stub_read_data  # type: ignore[assignment]

    proto._timeout = 1.0
    proto._read_timeout = 1.0
    proto._max_continuation_frames = 100

    with pytest.raises(OperationalError) as exc_info:
        await asyncio.wait_for(
            proto._read_continuation(deadline=999_999.0),
            timeout=1.0,
        )

    err = exc_info.value
    rendered = str(err)
    assert rendered.endswith(" to host-c:19003"), (
        f"Addr suffix dropped after display truncation: ...{rendered[-200:]!r}"
    )
    assert "[truncated," in rendered, (
        f"Expected truncation marker in display message: {rendered[:200]!r}..."
    )


@pytest.mark.asyncio
async def test_continuation_failure_arm_short_message_keeps_suffix() -> None:
    """Short FAILURE message: suffix is appended verbatim, no truncation."""
    from dqlitewire.exceptions import ServerFailure as _WireServerFailure

    proto = _make_proto_with_address("host-d:9000")

    class _StubDecoder:
        def decode_continuation(self) -> _WireServerFailure | None:
            raise _WireServerFailure(code=42, message="boom")

        def feed(self, data: bytes) -> None:
            pass

    proto._decoder = _StubDecoder()  # type: ignore[assignment]

    async def _stub_read_data(deadline: float) -> bytes:
        return b""

    proto._read_data = _stub_read_data  # type: ignore[assignment]

    proto._timeout = 1.0
    proto._read_timeout = 1.0
    proto._max_continuation_frames = 100

    with pytest.raises(OperationalError) as exc_info:
        await asyncio.wait_for(
            proto._read_continuation(deadline=999_999.0),
            timeout=1.0,
        )

    rendered = str(exc_info.value)
    assert rendered.endswith(" to host-d:9000")
    assert "[truncated," not in rendered
