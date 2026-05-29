"""allowlist_policy parses addresses via _parse_address so case-insensitivity, IPv6
bracket normalization, and port validation carry through to the redirect filter.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import allowlist_policy


def test_hostname_case_insensitive() -> None:
    policy = allowlist_policy(["Example.com:9001"])
    assert policy("example.com:9001") is True
    assert policy("EXAMPLE.COM:9001") is True


def test_hostname_case_in_listed_entry_normalized() -> None:
    """Two entries differing only in case dedupe to one (parser normalizes both)."""
    policy = allowlist_policy(["example.com:9001", "EXAMPLE.COM:9001"])
    assert policy("Example.Com:9001") is True
    assert policy("example.com:9002") is False


def test_ipv4_exact_match() -> None:
    policy = allowlist_policy(["127.0.0.1:9001"])
    assert policy("127.0.0.1:9001") is True
    assert policy("127.0.0.2:9001") is False


def test_ipv4_different_port_rejected() -> None:
    policy = allowlist_policy(["127.0.0.1:9001"])
    assert policy("127.0.0.1:9002") is False


# _parse_address rejects unbracketed IPv6.


def test_ipv6_bracketed_match() -> None:
    policy = allowlist_policy(["[::1]:9001"])
    assert policy("[::1]:9001") is True


def test_ipv6_different_bracketed_rejected() -> None:
    policy = allowlist_policy(["[::1]:9001"])
    assert policy("[::2]:9001") is False


def test_rejects_malformed_entry_at_construction() -> None:
    """A malformed entry raises at construction so a typo surfaces at config-load time."""
    with pytest.raises(ValueError):
        allowlist_policy(["not a valid address"])


def test_rejects_unbracketed_ipv6_at_construction() -> None:
    """Unbracketed IPv6 cannot be parsed unambiguously; require bracketed form."""
    with pytest.raises(ValueError):
        allowlist_policy(["::1:9001"])


def test_rejects_invalid_port_at_construction() -> None:
    with pytest.raises(ValueError):
        allowlist_policy(["host:abc"])


def test_rejects_malformed_runtime_address() -> None:
    """A malformed runtime address returns False, not raises, so a malicious server
    cannot crash the policy callback via a garbage redirect."""
    policy = allowlist_policy(["127.0.0.1:9001"])
    assert policy("not a valid address") is False


def test_rejects_unbracketed_ipv6_runtime() -> None:
    """A runtime unbracketed IPv6 is treated as a rejection rather than crashing."""
    policy = allowlist_policy(["[::1]:9001"])
    assert policy("::1:9001") is False


def test_accepts_iterable_input() -> None:
    """Construction accepts any iterable (list, set, generator, dict_keys)."""
    policy = allowlist_policy(addr for addr in ["[::1]:9001", "127.0.0.1:9001"])
    assert policy("[::1]:9001") is True
    assert policy("127.0.0.1:9001") is True


def test_empty_allowlist_rejects_all() -> None:
    policy = allowlist_policy([])
    assert policy("[::1]:9001") is False
    assert policy("127.0.0.1:9001") is False
