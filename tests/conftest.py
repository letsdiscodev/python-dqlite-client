"""Pytest configuration for dqlite-client tests."""

import contextlib
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqlitewire.constants import ValueType
from dqlitewire.messages import (
    DbResponse,
    LeaderResponse,
    ResultResponse,
    RowsResponse,
    WelcomeResponse,
)

# Add python-dqlite-dev's testlib (expected as a sibling checkout) to sys.path so
# the integration suite can import ``dqlitetestlib``; harmless when the sibling is absent.
_TESTLIB = Path(__file__).resolve().parent.parent.parent / "python-dqlite-dev" / "testlib"
if _TESTLIB.exists() and str(_TESTLIB) not in sys.path:
    sys.path.insert(0, str(_TESTLIB))

# Pytest 8+ requires ``pytest_plugins`` at the top-level conftest. Only register when
# the testlib resolved so unit-test-only consumers without the sibling repo are unaffected.
if _TESTLIB.exists():
    pytest_plugins = ["dqlitetestlib.fixtures"]


@pytest.fixture
def mock_reader() -> AsyncMock:
    reader = AsyncMock()
    return reader


@pytest.fixture
def mock_writer() -> MagicMock:
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


@pytest.fixture
def welcome_response() -> bytes:
    return WelcomeResponse(heartbeat_timeout=15000).encode()


@pytest.fixture
def leader_response() -> bytes:
    return LeaderResponse(node_id=1, address="localhost:9001").encode()


@pytest.fixture
def db_response() -> bytes:
    # Upstream always assigns id=0 to the first database on a fresh connection; mocks
    # MUST use db_id=0 to match (the client guards on it in open_database).
    return DbResponse(db_id=0).encode()


@pytest.fixture
def result_response() -> bytes:
    return ResultResponse(last_insert_id=1, rows_affected=1).encode()


@pytest.fixture
def rows_response() -> bytes:
    return RowsResponse(
        column_names=["id", "name"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[[1, "test"]],
        has_more=False,
    ).encode()


@pytest.fixture
async def connected_connection(
    mock_reader: AsyncMock,
    mock_writer: MagicMock,
    welcome_response: bytes,
    db_response: bytes,
) -> AsyncIterator[tuple[DqliteConnection, AsyncMock, MagicMock]]:
    """Connected DqliteConnection with mocked transport; yields (conn, reader, writer)."""
    mock_reader.read.side_effect = [welcome_response, db_response]
    conn = DqliteConnection("localhost:9001")
    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        await conn.connect()
    try:
        yield conn, mock_reader, mock_writer
    finally:
        with contextlib.suppress(Exception):
            await conn.close()
