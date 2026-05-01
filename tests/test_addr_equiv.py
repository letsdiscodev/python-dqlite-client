"""Pin: ``cluster._addr_equiv`` IPv6 bracket-tolerance and the
ValueError fallback for malformed addresses.

The function gates leader-redirect-trust: when
``_query_leader`` returns the redirect target, ``_addr_equiv``
decides whether the target is the same node as the
``node.address`` already in the store (skip authorization) or
a real redirect (run ``_check_redirect``). The canonical
``(host, port)`` tuple comparison must tolerate bracketed vs.
unbracketed IPv6 forms; malformed-on-both-sides must compare
equal via the literal-equality fallback so a hostile peer
cannot defeat the redirect-policy check by sending a non-
parseable address.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import _addr_equiv


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Identical canonical addresses.
        ("localhost:9001", "localhost:9001", True),
        ("[::1]:9001", "[::1]:9001", True),
        # Different hosts, same port.
        ("node-a:9001", "node-b:9001", False),
        # Same host, different ports.
        ("localhost:9001", "localhost:9002", False),
    ],
)
def test_addr_equiv_well_formed_addresses(a: str, b: str, expected: bool) -> None:
    assert _addr_equiv(a, b) is expected


def test_addr_equiv_unbracketed_ipv6_falls_back_to_literal_compare() -> None:
    """``_parse_address`` rejects unbracketed IPv6 — so the
    fallback compares literal strings. The two forms are NOT
    equivalent under ``_addr_equiv``; the redirect-trust gate
    runs the policy check on the unbracketed form. Pin this
    behaviour so a future relaxation of the parser would also
    update the docstring's "resolve to same tuple" claim."""
    assert _addr_equiv("[::1]:9001", "::1:9001") is False


def test_addr_equiv_falls_back_to_literal_equality_on_malformed() -> None:
    """When both sides are syntactically invalid, the parser
    raises ``ValueError`` and the fallback compares literal
    strings. Pin so a future refactor that changes the
    ValueError-fallback shape (e.g. returning False on parse
    error) can't silently flip the redirect-policy gate."""
    assert _addr_equiv("not-an-address", "not-an-address") is True
    assert _addr_equiv("not-an-address", "different-malformed") is False


def test_addr_equiv_one_side_malformed_returns_false() -> None:
    """One well-formed, one malformed → fallback to literal
    string comparison; the strings differ so result is False.
    Pin against a refactor that crashes on parse failure of one
    side."""
    assert _addr_equiv("localhost:9001", "garbage") is False
    assert _addr_equiv("garbage", "localhost:9001") is False
