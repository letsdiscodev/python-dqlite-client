"""High-level connection interface for dqlite."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol


def _parse_address(address: str) -> tuple[str, int]:
    """Parse a host:port address string, handling IPv6 brackets."""
    if address.startswith("["):
        # Bracketed IPv6: [host]:port
        if "]:" not in address:
            raise ValueError(
                f"Invalid IPv6 address format: expected '[host]:port', got {address!r}"
            )
        bracket_end = address.index("]")
        host = address[1:bracket_end]
        port_str = address[bracket_end + 2:]  # Skip ']:
    else:
        if ":" not in address:
            raise ValueError(
                f"Invalid address format: expected 'host:port', got {address!r}"
            )
        host, port_str = address.rsplit(":", 1)

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(
            f"Invalid port in address {address!r}: {port_str!r} is not a number"
        ) from None

    return host, port


class DqliteConnection:
    """High-level async connection to a dqlite database."""

    def __init__(
        self,
        address: str,
        *,
        database: str = "default",
        timeout: float = 10.0,
    ) -> None:
        """Initialize connection (does not connect yet).

        Args:
            address: Node address in "host:port" format
            database: Database name to open
            timeout: Connection timeout in seconds
        """
        self._address = address
        self._database = database
        self._timeout = timeout
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False

    @property
    def address(self) -> str:
        """Get the connection address."""
        return self._address

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._protocol is not None

    async def connect(self) -> None:
        """Establish connection to the database."""
        if self._protocol is not None:
            return

        host, port = _parse_address(self._address)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._timeout,
            )
        except TimeoutError as e:
            raise DqliteConnectionError(f"Connection to {self._address} timed out") from e
        except OSError as e:
            raise DqliteConnectionError(f"Failed to connect to {self._address}: {e}") from e

        self._protocol = DqliteProtocol(reader, writer, timeout=self._timeout)

        try:
            await self._protocol.handshake()
            self._db_id = await self._protocol.open_database(self._database)
        except Exception:
            self._protocol.close()
            self._protocol = None
            raise

    async def close(self) -> None:
        """Close the connection."""
        if self._protocol is not None:
            self._protocol.close()
            await self._protocol.wait_closed()
            self._protocol = None
            self._db_id = None

    async def __aenter__(self) -> "DqliteConnection":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _ensure_connected(self) -> tuple[DqliteProtocol, int]:
        """Ensure we're connected and return protocol and db_id."""
        if self._protocol is None or self._db_id is None:
            raise DqliteConnectionError("Not connected")
        return self._protocol, self._db_id

    def _invalidate(self) -> None:
        """Mark the connection as broken after an unrecoverable error."""
        self._protocol = None
        self._db_id = None

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement.

        Returns (last_insert_id, rows_affected).
        """
        protocol, db_id = self._ensure_connected()
        try:
            return await protocol.exec_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        protocol, db_id = self._ensure_connected()
        try:
            columns, rows = await protocol.query_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        """Execute a query and return results as list of lists."""
        protocol, db_id = self._ensure_connected()
        try:
            _, rows = await protocol.query_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        return rows

    async def fetchone(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        """Execute a query and return the first result."""
        results = await self.fetch(sql, params)
        return results[0] if results else None

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Execute a query and return the first column of the first row."""
        protocol, db_id = self._ensure_connected()
        try:
            _, rows = await protocol.query_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        if rows and rows[0]:
            return rows[0][0]
        return None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for transactions."""
        if self._in_transaction:
            raise OperationalError(
                0, "Nested transactions are not supported; use SAVEPOINT directly"
            )

        await self.execute("BEGIN")
        self._in_transaction = True

        try:
            yield
            await self.execute("COMMIT")
        except Exception:
            try:
                await self.execute("ROLLBACK")
            except Exception:
                pass  # Swallow rollback failure; original exception is more important
            raise
        finally:
            self._in_transaction = False
