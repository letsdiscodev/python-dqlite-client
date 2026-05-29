"""``_parse_address`` accepts a trailing-dot FQDN (RFC 1034 root-anchored) and
drops the dot, so rooted and unrooted forms share an allowlist key; a double
trailing dot stays invalid."""

from __future__ import annotations

import pytest

from dqliteclient.connection import _parse_address


@pytest.mark.parametrize(
    "addr",
    [
        "foo.example.com.:9001",
        "localhost.:9001",
        "single-label.:9001",
        "a.b.c.:9001",
    ],
)
def test_parse_address_accepts_trailing_dot_fqdn(addr: str) -> None:
    host, port = _parse_address(addr)
    assert not host.endswith(".")
    assert port == 9001


def test_parse_address_canonicalises_rooted_and_unrooted_to_same_tuple() -> None:
    """Root-anchored and bare forms canonicalise to the same tuple."""
    a = _parse_address("foo.example.com.:9001")
    b = _parse_address("foo.example.com:9001")
    assert a == b


def test_parse_address_rejects_double_trailing_dot() -> None:
    with pytest.raises(ValueError):
        _parse_address("foo..:9001")


def test_parse_address_rejects_lone_dot() -> None:
    with pytest.raises(ValueError):
        _parse_address(".:9001")
