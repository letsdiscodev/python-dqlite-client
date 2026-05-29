"""``parse_address`` rejects over-long inputs up front, reusing the wire-side
256-byte ``_MAX_ADDRESS_SIZE`` as SSOT so any address that parses also
round-trips through cluster discovery / redirect."""

from __future__ import annotations

import pytest

from dqliteclient.connection import _MAX_ADDRESS_LEN, parse_address


def test_oversized_address_rejected_with_bounded_message() -> None:
    huge = "a" * (_MAX_ADDRESS_LEN + 1) + ":9001"
    with pytest.raises(ValueError) as ei:
        parse_address(huge)
    # Bounded message must not interpolate the full input.
    assert len(str(ei.value)) < 1024


def test_oversized_address_message_describes_the_cap() -> None:
    huge = "a" * (1 << 20) + ":9001"
    with pytest.raises(ValueError, match="exceeds maximum"):
        parse_address(huge)


def test_at_cap_address_proceeds_to_normal_parse() -> None:
    """Below-cap address proceeds to the regular parse path."""
    host = ".".join(["a"] * 50)
    addr = f"{host}:9001"
    assert len(addr) < _MAX_ADDRESS_LEN
    out_host, out_port = parse_address(addr)
    assert out_port == 9001


def test_max_address_len_matches_wire_cap() -> None:
    """``_MAX_ADDRESS_LEN`` is the wire-side ``_MAX_ADDRESS_SIZE`` re-exported,
    not a parallel value."""
    from dqlitewire.messages.responses import _MAX_ADDRESS_SIZE

    assert _MAX_ADDRESS_LEN == _MAX_ADDRESS_SIZE


def test_address_above_wire_cap_rejected_with_actionable_message() -> None:
    """An above-cap address is rejected locally rather than dialing and later
    failing as ``DecodeError`` on every wire frame echoing it."""
    from dqlitewire.messages.responses import _MAX_ADDRESS_SIZE

    huge = "a" * _MAX_ADDRESS_SIZE + ":9001"
    with pytest.raises(ValueError, match="exceeds maximum"):
        parse_address(huge)


def test_non_string_input_rejected() -> None:
    with pytest.raises(ValueError, match="expected str"):
        parse_address(b"localhost:9001")  # type: ignore[arg-type]
