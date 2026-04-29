"""Pin: ``_parse_address`` accepts a trailing-dot FQDN (RFC 1034 §3.1
root-anchored notation). The hostname-label regex previously rejected
the form, contradicting RFC 3986 §3.2.2's ``reg-name`` permission.

Canonical form drops the trailing dot so two surface variants of the
same FQDN (rooted ``foo.example.com.`` and unrooted ``foo.example.com``)
canonicalise identically for allowlist comparisons.

A double trailing dot (``foo..``) remains invalid.
"""

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
    # Canonical form drops the trailing dot.
    assert not host.endswith(".")
    assert port == 9001


def test_parse_address_canonicalises_rooted_and_unrooted_to_same_tuple() -> None:
    """RFC 1034 root-anchored form and the bare form must canonicalise
    identically — allowlist policies hold one or the other; both
    surface variants must compare equal."""
    a = _parse_address("foo.example.com.:9001")
    b = _parse_address("foo.example.com:9001")
    assert a == b


def test_parse_address_rejects_double_trailing_dot() -> None:
    """``foo..`` is malformed (empty inner label) — must remain a
    ValueError."""
    with pytest.raises(ValueError):
        _parse_address("foo..:9001")


def test_parse_address_rejects_lone_dot() -> None:
    """A bare ``.`` is not a valid hostname."""
    with pytest.raises(ValueError):
        _parse_address(".:9001")
