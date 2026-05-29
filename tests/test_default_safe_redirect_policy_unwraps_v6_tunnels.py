"""default_safe_redirect_policy unwraps 6to4, Teredo, and ::ffff: IPv6 wrappers
before classifying, so a metadata/link-local v4 smuggled inside is rejected."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import default_safe_redirect_policy


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
