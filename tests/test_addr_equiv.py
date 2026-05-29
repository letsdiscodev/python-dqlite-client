"""Pin: ``cluster._addr_equiv`` IPv6 bracket-tolerance and its literal-equality
fallback for malformed addresses.

This gates leader-redirect-trust, so a malformed peer address must not be able
to defeat the redirect-policy check; it falls back to literal comparison.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import _addr_equiv


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("localhost:9001", "localhost:9001", True),
        ("[::1]:9001", "[::1]:9001", True),
        ("node-a:9001", "node-b:9001", False),
        ("localhost:9001", "localhost:9002", False),
    ],
)
def test_addr_equiv_well_formed_addresses(a: str, b: str, expected: bool) -> None:
    assert _addr_equiv(a, b) is expected


def test_addr_equiv_unbracketed_ipv6_falls_back_to_literal_compare() -> None:
    """Unbracketed IPv6 is rejected by the parser, so bracketed vs unbracketed
    forms compare unequal via the literal fallback (not the same tuple)."""
    assert _addr_equiv("[::1]:9001", "::1:9001") is False


def test_addr_equiv_falls_back_to_literal_equality_on_malformed() -> None:
    """Both sides malformed: the ValueError fallback compares literal strings."""
    assert _addr_equiv("not-an-address", "not-an-address") is True
    assert _addr_equiv("not-an-address", "different-malformed") is False


def test_addr_equiv_one_side_malformed_returns_false() -> None:
    """One side malformed: literal-string fallback yields False, no crash."""
    assert _addr_equiv("localhost:9001", "garbage") is False
    assert _addr_equiv("garbage", "localhost:9001") is False
