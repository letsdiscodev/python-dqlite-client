"""Pin: ``_read_continuation``'s server-FAILURE arm preserves the
peer-address attribution suffix in the resulting
``OperationalError.message`` even when the server's
``FailureResponse.message`` is at the wire-layer cap.

The eight other ``OperationalError`` raise sites in ``protocol.py``
go through ``self._failure_text(response)`` which pre-truncates the
body BEFORE appending the addr suffix, so the suffix survives the
``OperationalError._MAX_DISPLAY_MESSAGE`` (1024 codepoints) cap on
the exception's display ``message`` field.

The ``_read_continuation`` arm at lines 864-873 instead composes
``f"{e.message}{self._addr_suffix()}"`` and hands the un-pre-truncated
combined string to the constructor. A peer that emits a
``FailureResponse.message`` near the wire cap
(``_MAX_FAILURE_MESSAGE_SIZE = 64 KiB``) produces a string whose first
1024 codepoints are entirely consumed by the server-controlled
portion; the ``" to host:port"`` suffix is truncated off the visible
display.

This re-introduces the symptom that ISSUE-1063 (already in done/)
originally fixed at the non-continuation sites.
"""

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
    """Drive ``_read_continuation`` to the ``_WireServerFailure`` arm
    via the codec's mid-stream failure surface, with a server message
    near the wire cap. The resulting ``OperationalError.message`` must
    still end with the peer-address suffix.
    """
    from dqlitewire.exceptions import ServerFailure as _WireServerFailure

    proto = _make_proto_with_address("host-c:19003")

    # Stub the decoder + read-loop so the loop body re-raises a
    # ``_WireServerFailure`` with a 65000-char message — the same shape
    # the wire codec yields when a FAILURE frame arrives mid-ROWS-
    # stream. 65000 > OperationalError._MAX_DISPLAY_MESSAGE (1024) so
    # without pre-truncation the suffix is pushed off the display.
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

    # Initial continuation state — drive _read_continuation directly.
    with pytest.raises(OperationalError) as exc_info:
        await asyncio.wait_for(
            proto._read_continuation(deadline=999_999.0),
            timeout=1.0,
        )

    # The display message must end with the addr suffix even though the
    # server's raw message was 65000 chars.
    err = exc_info.value
    rendered = str(err)
    assert rendered.endswith(" to host-c:19003"), (
        f"Addr suffix dropped from continuation FAILURE error after "
        f"display truncation: ...{rendered[-200:]!r}"
    )
    # Defensive cap on the body still applies.
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
