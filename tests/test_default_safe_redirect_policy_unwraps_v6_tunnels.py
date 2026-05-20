"""Pin: ``default_safe_redirect_policy`` unwraps 6to4 (``2002::/16``)
and Teredo (``2001::/32``) IPv6 encapsulations before classifying, so a
metadata-service IPv4 (or any link-local v4) smuggled inside the
wrapper is correctly rejected. The existing ``::ffff:<v4>`` unwrap is
preserved.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import default_safe_redirect_policy


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254:9001",  # bare AWS / GCP metadata
        "[::ffff:169.254.169.254]:9001",  # IPv4-mapped
        "[2002:a9fe:a9fe::1]:9001",  # 6to4 wrap of 169.254.169.254
    ],
)
def test_default_policy_blocks_metadata_in_all_v6_tunnels(address: str) -> None:
    policy = default_safe_redirect_policy(include_rfc1918=False, include_loopback=False)
    assert policy(address) is False


def test_default_policy_blocks_teredo_wrapped_metadata() -> None:
    """Teredo encapsulates a client v4 as the last 32 bits (XOR'd
    against ``0xffffffff``). The Python ``ipaddress.IPv6Address.teredo``
    attribute unwraps automatically.
    """
    # Teredo prefix 2001::/32 + 32 bits server + 16 bits flags
    # + 16 bits port + 32 bits client v4 (XOR'd). The stdlib parses
    # this for us; we just need a valid Teredo address whose client
    # v4 is the metadata IP. 169.254.169.254 XOR 0xffffffff
    # = 0x56012956 = 86.1.41.86 — so construct:
    #   2001:0:<server>:0:0:0:<client-xored>
    # via the stdlib's own helper to avoid bit-fiddling here.
    import ipaddress

    server_v4 = ipaddress.IPv4Address("65.55.158.118")  # arbitrary; stdlib doc example
    client_v4 = ipaddress.IPv4Address("169.254.169.254")
    # Build the v6 address by hand:
    server_int = int(server_v4)
    client_xored = int(client_v4) ^ 0xFFFFFFFF
    teredo_int = (0x2001 << 112) | (server_int << 64) | client_xored
    teredo_v6 = ipaddress.IPv6Address(teredo_int)
    # Sanity: the stdlib's teredo attribute unwraps to our client v4.
    assert teredo_v6.teredo is not None
    assert teredo_v6.teredo[1] == client_v4
    policy = default_safe_redirect_policy(include_rfc1918=False, include_loopback=False)
    assert policy(f"[{teredo_v6}]:9001") is False
