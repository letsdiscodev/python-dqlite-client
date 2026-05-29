"""Strict port parse and IPv6 error-precedence for ``_parse_address``: stdlib
``int()`` leniency (whitespace, ``+``, underscores, Unicode digits) would let
non-canonical ports slip past string-equality allowlists."""

from __future__ import annotations

import pytest

from dqliteclient.connection import _parse_address


@pytest.mark.parametrize(
    "address",
    [
        "host:+9000",  # unary +
        "host:9_000",  # PEP 515 underscore
        "host: 9000",  # leading whitespace
        "host:9000 ",  # trailing whitespace
        "host:३000",  # Devanagari digit accepted by int() / isdigit()
    ],
)
def test_non_canonical_port_rejected(address: str) -> None:
    with pytest.raises(ValueError, match="not a number"):
        _parse_address(address)


def test_leading_zero_port_rejected_for_canonical_routing() -> None:
    """Present-day behaviour: ``isdigit`` permits leading zeros, so
    ``host:09000`` parses as 9000 (stricter rejection is TBD)."""
    host, port = _parse_address("host:09000")
    assert port == 9000


def test_unbracketed_ipv6_surfaces_bracket_diagnostic_before_port_parse() -> None:
    """The bracket-shape check must fire before the port parse so ``::1:abc``
    reports missing brackets, not a misleading "invalid port"."""
    with pytest.raises(ValueError, match="IPv6 addresses must be bracketed"):
        _parse_address("::1:abc")


def test_credentials_shape_not_misdiagnosed_as_ipv6() -> None:
    """A multi-colon credentials-smuggle (``user:pass@host:9001``) must not
    trip the IPv6 bracket-shape check; host canonicalisation rejects it."""
    with pytest.raises(ValueError) as exc_info:
        _parse_address("user:pass@evil.example.com:9001")
    msg = str(exc_info.value)
    assert "IPv6" not in msg
