"""``default_safe_redirect_policy`` rejects link-local + loopback by
default; RFC 1918 stays accepted because most production dqlite
clusters live on private subnets.

Link-local rejection is the canonical SSRF defence — the
AWS / GCP metadata IP ``169.254.169.254`` is the most common
target.
"""

import pytest

from dqliteclient import default_safe_redirect_policy


@pytest.fixture
def default_policy():
    return default_safe_redirect_policy()


def test_link_local_rejected(default_policy) -> None:
    """The AWS / GCP metadata endpoint MUST be rejected."""
    assert default_policy("169.254.169.254:80") is False


def test_link_local_v6_rejected(default_policy) -> None:
    """fe80::/10 is link-local IPv6."""
    assert default_policy("[fe80::1]:9001") is False


def test_loopback_rejected_by_default(default_policy) -> None:
    """Loopback is rare in production cluster topologies; reject by
    default but allow operators to opt in for tests."""
    assert default_policy("127.0.0.1:9001") is False


def test_loopback_accepted_with_include_loopback() -> None:
    """Test environments binding to loopback opt in via the kwarg."""
    policy = default_safe_redirect_policy(include_loopback=True)
    assert policy("127.0.0.1:9001") is True


def test_rfc1918_accepted_by_default(default_policy) -> None:
    """Most production dqlite clusters live on RFC 1918 subnets;
    rejecting by default would be too restrictive."""
    assert default_policy("10.0.0.5:9001") is True
    assert default_policy("172.16.5.10:9001") is True
    assert default_policy("192.168.1.20:9001") is True


def test_rfc1918_rejected_when_disabled() -> None:
    """Internet-facing clusters can opt out."""
    policy = default_safe_redirect_policy(include_rfc1918=False)
    assert policy("10.0.0.5:9001") is False


def test_public_ip_accepted(default_policy) -> None:
    assert default_policy("203.0.113.5:9001") is True


def test_hostname_passes_through(default_policy) -> None:
    """DNS hostnames are not classified by the IP-based filter; rely
    on the operator's DNS layer / explicit allowlist."""
    assert default_policy("api.example.com:9001") is True


def test_malformed_address_rejected(default_policy) -> None:
    """Hostile input cannot crash the policy."""
    assert default_policy("not-a-valid-host-port") is False
    assert default_policy("") is False
    assert default_policy("[malformed") is False


def test_top_level_export() -> None:
    """The factory is reachable from the package top level."""
    import dqliteclient

    assert "default_safe_redirect_policy" in dqliteclient.__all__
    assert dqliteclient.default_safe_redirect_policy is default_safe_redirect_policy
