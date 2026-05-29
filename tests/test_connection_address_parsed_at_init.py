"""Malformed addresses raise ``ValueError`` at construction, not inside
``connect()``: otherwise SA's ``is_disconnect`` relabels a permanent DSN typo as
a retryable disconnect and the caller pays the full connect-timeout first."""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.parametrize(
    "bad_address",
    [
        "no-colon-here",
        "host:not-a-port",
        "host:99999",
        "[::1:9001",  # unclosed IPv6 bracket
        "user@host:9001",  # credentials-like
    ],
)
def test_malformed_address_fails_at_construction(bad_address: str) -> None:
    with pytest.raises(ValueError):
        DqliteConnection(bad_address)


def test_well_formed_address_exposes_cached_host_port() -> None:
    conn = DqliteConnection("127.0.0.1:9001")
    # Parse result cached without calling connect().
    assert conn._host == "127.0.0.1"
    assert conn._port == 9001
    assert conn._address == "127.0.0.1:9001"
