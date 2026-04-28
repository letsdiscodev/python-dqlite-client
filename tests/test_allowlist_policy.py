"""``allowlist_policy`` must use the same address-parsing logic as
the rest of the cluster module so the equivalence properties of
:func:`_parse_address` (hostname case-insensitivity, IPv6 bracket
normalization, port validation) carry through to the redirect
filter.

The naive ``addr in frozenset(addresses)`` form would reject a
case-different hostname (``"Example.com:9001"`` vs
``"example.com:9001"``) and would silently accept malformed
allow-list entries that no real server could ever produce.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import allowlist_policy

# Hostname casing — `_parse_address` lower-cases the host so callers
# do not have to think about it. The naive raw-string match did not
# do this.


def test_hostname_case_insensitive() -> None:
    policy = allowlist_policy(["Example.com:9001"])
    assert policy("example.com:9001") is True
    assert policy("EXAMPLE.COM:9001") is True


def test_hostname_case_in_listed_entry_normalized() -> None:
    """Two entries that differ only in case dedupe to one (parser
    normalizes both)."""
    policy = allowlist_policy(["example.com:9001", "EXAMPLE.COM:9001"])
    assert policy("Example.Com:9001") is True
    assert policy("example.com:9002") is False


# IPv4 round-trips


def test_ipv4_exact_match() -> None:
    policy = allowlist_policy(["127.0.0.1:9001"])
    assert policy("127.0.0.1:9001") is True
    assert policy("127.0.0.2:9001") is False


def test_ipv4_different_port_rejected() -> None:
    policy = allowlist_policy(["127.0.0.1:9001"])
    assert policy("127.0.0.1:9002") is False


# IPv6 (bracketed only — ``_parse_address`` rejects unbracketed IPv6)


def test_ipv6_bracketed_match() -> None:
    policy = allowlist_policy(["[::1]:9001"])
    assert policy("[::1]:9001") is True


def test_ipv6_different_bracketed_rejected() -> None:
    policy = allowlist_policy(["[::1]:9001"])
    assert policy("[::2]:9001") is False


# Construction-time validation


def test_rejects_malformed_entry_at_construction() -> None:
    """A malformed allow-list entry raises at construction so the
    operator's typo surfaces at config-load time, not as a silent
    rejection of a future legitimate redirect."""
    with pytest.raises(ValueError):
        allowlist_policy(["not a valid address"])


def test_rejects_unbracketed_ipv6_at_construction() -> None:
    """Unbracketed IPv6 cannot be parsed unambiguously; reject so
    the operator must spell it bracketed (matching the wire and the
    rest of the client surface)."""
    with pytest.raises(ValueError):
        allowlist_policy(["::1:9001"])


def test_rejects_invalid_port_at_construction() -> None:
    with pytest.raises(ValueError):
        allowlist_policy(["host:abc"])


# Runtime malformed addresses


def test_rejects_malformed_runtime_address() -> None:
    """A malformed runtime address returns False rather than raising,
    so a malicious server cannot crash the policy callback by
    sending garbage in a redirect response."""
    policy = allowlist_policy(["127.0.0.1:9001"])
    assert policy("not a valid address") is False


def test_rejects_unbracketed_ipv6_runtime() -> None:
    """A runtime unbracketed IPv6 cannot be parsed; treat as a
    rejection rather than crashing."""
    policy = allowlist_policy(["[::1]:9001"])
    assert policy("::1:9001") is False


# Iterable shapes


def test_accepts_iterable_input() -> None:
    """Construction accepts any iterable — list, set, generator,
    dict_keys."""
    policy = allowlist_policy(addr for addr in ["[::1]:9001", "127.0.0.1:9001"])
    assert policy("[::1]:9001") is True
    assert policy("127.0.0.1:9001") is True


def test_empty_allowlist_rejects_all() -> None:
    policy = allowlist_policy([])
    assert policy("[::1]:9001") is False
    assert policy("127.0.0.1:9001") is False
