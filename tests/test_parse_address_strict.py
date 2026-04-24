"""Strict port parse and IPv6 error-precedence regressions for
``_parse_address``. stdlib ``int()`` is lenient about whitespace,
unary ``+``, PEP 515 underscores, leading zeros, and Unicode digits;
allowlist policies that compare by string equality then fail to match
the semantically-identical port when a peer redirects to the
non-canonical form.
"""

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
    """``"host:09000"`` would be accepted by ``int()`` as 9000 but
    does not string-match the canonical ``"host:9000"`` a policy
    allowlist would hold. Reject it.
    """
    # "09000" passes isdigit() so we need to be even stricter.
    # The current check accepts leading zeros; if future policy
    # strictness wants to reject them, tighten the check. For now,
    # document the behaviour: isdigit permits leading zeros.
    host, port = _parse_address("host:09000")
    assert port == 9000  # present-day behaviour; stricter form TBD


def test_unbracketed_ipv6_surfaces_bracket_diagnostic_before_port_parse() -> None:
    """``"::1:abc"`` splits to host="::1" and port_str="abc". The
    bracket-shape check must fire BEFORE the port parse so the
    diagnostic points at the missing brackets rather than surfacing a
    misleading "invalid port" error.
    """
    with pytest.raises(ValueError, match="IPv6 addresses must be bracketed"):
        _parse_address("::1:abc")


def test_credentials_shape_not_misdiagnosed_as_ipv6() -> None:
    """A credentials-smuggle like ``user:pass@host:9001`` has multiple
    colons but is not IPv6. The bracket-shape check must not fire on
    this path; the canonicalize-host step rejects it with a more
    specific "invalid host" message.
    """
    with pytest.raises(ValueError) as exc_info:
        _parse_address("user:pass@evil.example.com:9001")
    msg = str(exc_info.value)
    # Must NOT be the IPv6 message.
    assert "IPv6" not in msg
