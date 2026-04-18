"""Validation of the ``max_total_rows`` constructor parameter."""

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.pool import ConnectionPool
from dqliteclient.protocol import DqliteProtocol, _validate_max_total_rows


class TestValidator:
    def test_none_allowed(self) -> None:
        assert _validate_max_total_rows(None) is None

    def test_positive_int_allowed(self) -> None:
        assert _validate_max_total_rows(1) == 1
        assert _validate_max_total_rows(10_000_000) == 10_000_000

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_total_rows must be > 0"):
            _validate_max_total_rows(0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_total_rows must be > 0"):
            _validate_max_total_rows(-1)

    def test_float_rejected(self) -> None:
        with pytest.raises(TypeError, match="max_total_rows must be int or None"):
            _validate_max_total_rows(1.5)  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        # True is technically int, but PEP-489-style APIs rightly reject it.
        with pytest.raises(TypeError, match="max_total_rows must be int or None"):
            _validate_max_total_rows(True)  # type: ignore[arg-type]

    def test_string_rejected(self) -> None:
        with pytest.raises(TypeError, match="max_total_rows must be int or None"):
            _validate_max_total_rows("100")  # type: ignore[arg-type]


class TestConstructorValidation:
    def test_dqlite_connection_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            DqliteConnection("localhost:19001", max_total_rows=0)

    def test_dqlite_connection_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            DqliteConnection("localhost:19001", max_total_rows=-5)

    def test_dqlite_connection_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            DqliteConnection("localhost:19001", max_total_rows=True)  # type: ignore[arg-type]

    def test_dqlite_connection_none_allowed(self) -> None:
        conn = DqliteConnection("localhost:19001", max_total_rows=None)
        assert conn._max_total_rows is None

    def test_pool_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            ConnectionPool(addresses=["localhost:19001"], max_total_rows=0)

    def test_pool_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            ConnectionPool(addresses=["localhost:19001"], max_total_rows=-1)

    @pytest.mark.asyncio
    async def test_protocol_zero_rejected(self) -> None:
        reader = asyncio.StreamReader()
        writer = _DummyWriter()
        with pytest.raises(ValueError):
            DqliteProtocol(reader, writer, max_total_rows=0)  # type: ignore[arg-type]


class _DummyWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass
