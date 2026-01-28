"""Pytest configuration for dqlite-client tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.messages import (
    DbResponse,
    LeaderResponse,
    ResultResponse,
    RowsResponse,
    WelcomeResponse,
)


@pytest.fixture
def mock_reader() -> AsyncMock:
    """Create a mock StreamReader."""
    reader = AsyncMock()
    return reader


@pytest.fixture
def mock_writer() -> MagicMock:
    """Create a mock StreamWriter."""
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


@pytest.fixture
def welcome_response() -> bytes:
    """Create encoded WelcomeResponse."""
    return WelcomeResponse(heartbeat_timeout=15000).encode()


@pytest.fixture
def leader_response() -> bytes:
    """Create encoded LeaderResponse."""
    return LeaderResponse(node_id=1, address="localhost:9001").encode()


@pytest.fixture
def db_response() -> bytes:
    """Create encoded DbResponse."""
    return DbResponse(db_id=1).encode()


@pytest.fixture
def result_response() -> bytes:
    """Create encoded ResultResponse."""
    return ResultResponse(last_insert_id=1, rows_affected=1).encode()


@pytest.fixture
def rows_response() -> bytes:
    """Create encoded RowsResponse."""
    return RowsResponse(
        column_names=["id", "name"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[[1, "test"]],
        has_more=False,
    ).encode()
