"""Coverage-gap tests for reachable-but-untested branches in dqliteclient."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection, _parse_address
from dqliteclient.node_store import MemoryNodeStore


class TestParseAddressIPv6ZoneIdRejection:
    """``_canonicalize_host`` rejects malformed zone identifiers on bracketed IPv6."""

    def test_empty_zone_rejected(self) -> None:
        with pytest.raises(ValueError):
            _parse_address("[fe80::1%]:9001")

    def test_invalid_zone_character_space_rejected(self) -> None:
        with pytest.raises(ValueError):
            _parse_address("[fe80::1%foo bar]:9001")

    def test_invalid_zone_character_slash_rejected(self) -> None:
        with pytest.raises(ValueError):
            _parse_address("[fe80::1%foo/bar]:9001")


class TestMemoryNodeStoreInitRejections:
    def test_rejects_non_string_entry(self) -> None:
        with pytest.raises(TypeError):
            MemoryNodeStore([42])  # type: ignore[list-item]

    def test_rejects_empty_string_entry(self) -> None:
        with pytest.raises(ValueError):
            MemoryNodeStore([""])

    def test_dedupes_duplicate_addresses(self) -> None:
        async def _drive() -> None:
            store = MemoryNodeStore(["localhost:9001", "localhost:9001", "localhost:9002"])
            nodes = await store.get_nodes()
            addrs = [n.address for n in nodes]
            assert addrs.count("localhost:9001") == 1
            assert addrs.count("localhost:9002") == 1

        asyncio.run(_drive())


class TestConnectAlreadyConnectedFastPath:
    @pytest.mark.asyncio
    async def test_connect_returns_immediately_when_already_connected(self) -> None:
        """``connect()`` is idempotent: a second call when already connected is a no-op."""
        from unittest.mock import MagicMock

        conn = DqliteConnection("127.0.0.1:9999", timeout=0.5)
        # Fast-path triggers on ``self._protocol is not None`` (identity only).
        sentinel_protocol = MagicMock()
        conn._protocol = sentinel_protocol
        conn._db_id = 1

        await conn.connect()

        assert conn._protocol is sentinel_protocol
        assert conn._db_id == 1
