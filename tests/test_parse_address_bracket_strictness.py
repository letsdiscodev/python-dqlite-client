"""Pin: bracket syntax in ``_parse_address`` is reserved for IPv6
literals (RFC 3986 §3.2.2). Bracketed IPv4 / hostname / empty
contents must be rejected.

Pin: IPv6 zone identifiers in bracketed form percent-decode per
RFC 6874 (`%25` escapes the literal `%`). Both surface variants of
the same logical zone canonicalise to the same tuple.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import _parse_address


class TestBracketedNonIpv6Rejected:
    @pytest.mark.parametrize(
        "addr",
        [
            "[127.0.0.1]:9001",
            "[localhost]:9001",
            "[example.com]:9001",
            "[]:9001",
            "[ ]:9001",
        ],
    )
    def test_bracketed_non_ipv6_raises(self, addr: str) -> None:
        with pytest.raises(ValueError):
            _parse_address(addr)

    def test_real_ipv6_still_parses(self) -> None:
        assert _parse_address("[::1]:9001") == ("::1", 9001)
        assert _parse_address("[fe80::1]:9001") == ("fe80::1", 9001)
        assert _parse_address("[2001:db8::1]:9001") == ("2001:db8::1", 9001)


class TestIpv6ZoneIdPercentEncoding:
    def test_zone_id_decoded_canonicalises_two_surface_forms(self) -> None:
        """RFC 6874: the URI-form ``%25eth0`` and the application-form
        ``%eth0`` must canonicalise to the same tuple so allowlist
        policies match either surface variant."""
        a = _parse_address("[fe80::1%eth0]:9001")
        b = _parse_address("[fe80::1%25eth0]:9001")
        assert a == b
        assert "%eth0" in a[0]
        # Post-decode form must NOT contain the URI-encoded sequence.
        assert "%25" not in a[0]

    def test_unencoded_zone_id_still_works(self) -> None:
        host, port = _parse_address("[fe80::1%eth0]:9001")
        assert host == "fe80::1%eth0"
        assert port == 9001
