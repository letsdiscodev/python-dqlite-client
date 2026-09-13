"""Redirect policies: default_safe_redirect_policy, allowlist_policy and bad redirect addresses."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy, default_safe_redirect_policy
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


@pytest.fixture
def default_policy():
    return default_safe_redirect_policy()


def test_link_local_rejected(default_policy) -> None:
    """The cloud metadata endpoint must be rejected."""
    assert default_policy("169.254.169.254:80") is False


def test_link_local_v6_rejected(default_policy) -> None:
    """fe80::/10 link-local IPv6."""
    assert default_policy("[fe80::1]:9001") is False


def test_loopback_rejected_by_default(default_policy) -> None:
    """Loopback rejected by default."""
    assert default_policy("127.0.0.1:9001") is False


def test_loopback_accepted_with_include_loopback() -> None:
    """include_loopback=True opts loopback in."""
    policy = default_safe_redirect_policy(include_loopback=True)
    assert policy("127.0.0.1:9001") is True


def test_rfc1918_accepted_by_default(default_policy) -> None:
    """RFC 1918 accepted by default."""
    assert default_policy("10.0.0.5:9001") is True
    assert default_policy("172.16.5.10:9001") is True
    assert default_policy("192.168.1.20:9001") is True


def test_rfc1918_rejected_when_disabled() -> None:
    """include_rfc1918=False opts RFC 1918 out."""
    policy = default_safe_redirect_policy(include_rfc1918=False)
    assert policy("10.0.0.5:9001") is False


def test_public_ip_accepted(default_policy) -> None:
    assert default_policy("203.0.113.5:9001") is True


def test_hostname_passes_through(default_policy) -> None:
    """DNS hostnames are not classified by the IP-based filter."""
    assert default_policy("api.example.com:9001") is True


def test_malformed_address_rejected(default_policy) -> None:
    """Malformed input rejected, never crashes."""
    assert default_policy("not-a-valid-host-port") is False
    assert default_policy("") is False
    assert default_policy("[malformed") is False


def test_top_level_export() -> None:
    import dqliteclient

    assert "default_safe_redirect_policy" in dqliteclient.__all__
    assert dqliteclient.default_safe_redirect_policy is default_safe_redirect_policy


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254:9001",  # bare metadata
        "[::ffff:169.254.169.254]:9001",  # IPv4-mapped
        "[2002:a9fe:a9fe::1]:9001",  # 6to4 wrap of 169.254.169.254
    ],
)
def test_default_policy_blocks_metadata_in_all_v6_tunnels(address: str) -> None:
    policy = default_safe_redirect_policy(include_rfc1918=False, include_loopback=False)
    assert policy(address) is False


def test_default_policy_blocks_teredo_wrapped_metadata() -> None:
    """Teredo encodes the client v4 as the last 32 bits XOR'd with 0xffffffff;
    ipaddress.IPv6Address.teredo unwraps it."""
    import ipaddress

    server_v4 = ipaddress.IPv4Address("65.55.158.118")  # arbitrary; stdlib doc example
    client_v4 = ipaddress.IPv4Address("169.254.169.254")
    server_int = int(server_v4)
    client_xored = int(client_v4) ^ 0xFFFFFFFF
    teredo_int = (0x2001 << 112) | (server_int << 64) | client_xored
    teredo_v6 = ipaddress.IPv6Address(teredo_int)
    assert teredo_v6.teredo is not None
    assert teredo_v6.teredo[1] == client_v4
    policy = default_safe_redirect_policy(include_rfc1918=False, include_loopback=False)
    assert policy(f"[{teredo_v6}]:9001") is False


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


@pytest.mark.asyncio
async def test_server_redirect_to_invalid_address_raises_cluster_policy_error() -> None:
    """An address with control bytes fails _canonicalize_host; the wrap holds."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        return "host\nnewline:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_server_redirect_to_non_ascii_address_raises_cluster_policy_error() -> None:
    """Non-ASCII host (IDN) is rejected; the wrap holds."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        return "résumé.example.com:9001"

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_server_redirect_to_oversized_hostname_raises_cluster_policy_error() -> None:
    """Hostname > 253 chars exceeds the DNS limit; the wrap holds."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=0.1)

    oversize = "a" * 260 + ".example.com:9001"

    async def _fake_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        return oversize

    with (
        patch.object(client, "find_leader", new=_fake_find_leader),
        pytest.raises(ClusterPolicyError, match="invalid leader address"),
    ):
        await client.connect()
