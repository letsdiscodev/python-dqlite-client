"""``parse_address`` rejects unspecified / multicast / reserved IP literals
(invalid TCP destinations). IPv4-mapped IPv6 is unwrapped first, since
``is_unspecified`` on the IPv6 wrapper misses an unspecified embedded IPv4."""

from __future__ import annotations

import pytest

from dqliteclient.connection import _parse_address


@pytest.mark.parametrize(
    "address",
    [
        "0.0.0.0:9001",
        "[::]:9001",
        "[0:0:0:0:0:0:0:0]:9001",
        "[::ffff:0.0.0.0]:9001",
    ],
    ids=[
        "ipv4-unspecified",
        "ipv6-unspecified-shorthand",
        "ipv6-unspecified-explicit",
        "ipv4-mapped-unspecified",
    ],
)
def test_parse_address_rejects_unspecified_ip_literals(address: str) -> None:
    with pytest.raises(ValueError, match="unspecified"):
        _parse_address(address)


@pytest.mark.parametrize(
    "address",
    [
        "224.0.0.1:9001",
        "239.255.255.255:9001",
        "[ff02::1]:9001",
    ],
    ids=["ipv4-multicast-low", "ipv4-multicast-high", "ipv6-multicast"],
)
def test_parse_address_rejects_multicast_ip_literals(address: str) -> None:
    with pytest.raises(ValueError, match="multicast"):
        _parse_address(address)


@pytest.mark.parametrize(
    "address",
    [
        "240.0.0.1:9001",
    ],
    ids=["ipv4-reserved"],
)
def test_parse_address_rejects_reserved_ip_literals(address: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        _parse_address(address)


def test_parse_address_accepts_normal_ipv4() -> None:
    host, port = _parse_address("127.0.0.1:9001")
    assert host == "127.0.0.1"
    assert port == 9001


def test_parse_address_accepts_normal_ipv6() -> None:
    host, port = _parse_address("[::1]:9001")
    assert host == "::1"
    assert port == 9001


def test_parse_address_canonicalises_ipv4_mapped_ipv6_to_v4() -> None:
    """RFC 4291 §2.5.5.2: IPv4-mapped IPv6 canonicalises to the dotted-quad,
    so it shares a tuple with the bare IPv4 form (one allowlist key)."""
    v4_form = _parse_address("127.0.0.1:9001")
    mapped_form = _parse_address("[::ffff:127.0.0.1]:9001")
    assert v4_form == mapped_form
    assert v4_form == ("127.0.0.1", 9001)


def test_parse_address_canonicalises_ipv4_mapped_ipv6_loopback_explicit() -> None:
    """The hex form ``::ffff:7f00:1`` decomposes to the same IPv4 dotted-quad."""
    v4_form = _parse_address("127.0.0.1:9001")
    hex_mapped = _parse_address("[::ffff:7f00:1]:9001")
    assert v4_form == hex_mapped


def test_parse_address_accepts_normal_hostname() -> None:
    host, port = _parse_address("example.com:9001")
    assert host == "example.com"
    assert port == 9001
