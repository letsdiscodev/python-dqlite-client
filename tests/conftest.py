"""Pytest configuration for dqlite-client tests."""

import sys
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

# Add python-dqlite-dev's testlib to sys.path so tests (in particular
# the integration suite) can import shared utilities from
# ``dqlitetestlib``. ``python-dqlite-dev`` is expected as a sibling of
# this checkout — see ``python-dqlite-dev/testlib/README.md``. The
# insertion is harmless when the sibling repo is absent.
_TESTLIB = Path(__file__).resolve().parent.parent.parent / "python-dqlite-dev" / "testlib"
if _TESTLIB.exists() and str(_TESTLIB) not in sys.path:
    sys.path.insert(0, str(_TESTLIB))

# Pytest 8+ requires ``pytest_plugins`` at the top-level conftest.
# Only register the testlib's fixtures plugin when the path resolved
# so consumers running unit tests without the sibling repo see no
# difference.
if _TESTLIB.exists():
    pytest_plugins = ["dqlitetestlib.fixtures"]


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


@pytest.fixture
async def connected_connection(
    mock_reader: AsyncMock,
    mock_writer: MagicMock,
    welcome_response: bytes,
    db_response: bytes,
) -> tuple[DqliteConnection, AsyncMock, MagicMock]:
    """A DqliteConnection that is already connected with mocked transport.

    Returns (conn, mock_reader, mock_writer) so tests can configure
    additional response data on mock_reader.
    """
    mock_reader.read.side_effect = [welcome_response, db_response]
    conn = DqliteConnection("localhost:9001")
    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        await conn.connect()
    return conn, mock_reader, mock_writer
