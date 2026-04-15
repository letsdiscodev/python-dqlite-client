"""High-level connection interface for dqlite."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dqliteclient.exceptions import (
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.protocol import DqliteProtocol

# dqlite error codes that indicate a leader change (SQLite extended error codes)
# SQLITE_IOERR_NOT_LEADER = SQLITE_IOERR | (40 << 8) = 10250
# SQLITE_IOERR_LEADERSHIP_LOST = SQLITE_IOERR | (41 << 8) = 10506
_LEADER_ERROR_CODES = {10250, 10506}


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
        port_str = address[bracket_end + 2 :]  # Skip ']:
    else:
        if ":" not in address:
            raise ValueError(f"Invalid address format: expected 'host:port', got {address!r}")
        host, port_str = address.rsplit(":", 1)

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(
            f"Invalid port in address {address!r}: {port_str!r} is not a number"
        ) from None

    return host, port


class DqliteConnection:
    """High-level async connection to a dqlite database.

    Thread safety: this class is NOT thread-safe. All operations must be
    performed within a single asyncio event loop. Do not share instances
    across OS threads or event loops. To submit work from other threads,
    use ``asyncio.run_coroutine_threadsafe()`` — the coroutines execute
    safely in the event loop thread. Free-threaded Python (no-GIL) is
    not supported.
    """

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
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self._address = address
        self._database = database
        self._timeout = timeout
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False
        self._in_use = False
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tx_owner: asyncio.Task[Any] | None = None

    @property
    def address(self) -> str:
        """Get the connection address."""
        return self._address

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._protocol is not None

    @property
    def in_transaction(self) -> bool:
        """Check if a transaction is active."""
        return self._in_transaction

    async def connect(self) -> None:
        """Establish connection to the database."""
        self._check_in_use()
        if self._protocol is not None:
            return

        self._bound_loop = asyncio.get_running_loop()
        self._in_use = True
        try:
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
            except BaseException:
                self._protocol.close()
                self._protocol = None
                raise
        finally:
            self._in_use = False

    async def close(self) -> None:
        """Close the connection."""
        self._check_in_use()
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

    def _check_in_use(self) -> None:
        """Raise on misuse: wrong event loop or concurrent coroutine access."""
        if self._bound_loop is not None:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                raise InterfaceError(
                    "DqliteConnection must be used from within an async context."
                ) from None
            if current_loop is not self._bound_loop:
                raise InterfaceError(
                    "DqliteConnection is bound to a different event loop. "
                    "Do not share connections across event loops or OS threads."
                )
        if self._in_use:
            raise InterfaceError(
                "Cannot perform operation: another operation is in progress on this "
                "connection. DqliteConnection does not support concurrent coroutine "
                "access. Use a ConnectionPool to manage multiple concurrent operations."
            )
        if self._in_transaction and self._tx_owner is not None:
            current = asyncio.current_task()
            if current is not self._tx_owner:
                raise InterfaceError(
                    "Cannot perform operation: connection is in a transaction owned "
                    "by another task. Each task should use its own connection from "
                    "the pool."
                )

    def _invalidate(self) -> None:
        """Mark the connection as broken after an unrecoverable error."""
        if self._protocol is not None:
            # Connection may already be broken; suppress close errors
            with contextlib.suppress(Exception):
                self._protocol.close()
        self._protocol = None
        self._db_id = None

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement.

        Returns (last_insert_id, rows_affected).
        """
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        self._in_use = True
        try:
            return await protocol.exec_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        except OperationalError as e:
            if e.code in _LEADER_ERROR_CODES:
                self._invalidate()
            raise
        except BaseException:
            self._invalidate()
            raise
        finally:
            self._in_use = False

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        self._in_use = True
        try:
            columns, rows = await protocol.query_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        except OperationalError as e:
            if e.code in _LEADER_ERROR_CODES:
                self._invalidate()
            raise
        except BaseException:
            self._invalidate()
            raise
        finally:
            self._in_use = False
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        """Execute a query and return results as list of lists."""
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        self._in_use = True
        try:
            _, rows = await protocol.query_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        except OperationalError as e:
            if e.code in _LEADER_ERROR_CODES:
                self._invalidate()
            raise
        except BaseException:
            self._invalidate()
            raise
        finally:
            self._in_use = False
        return rows

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute a query and return the first result."""
        results = await self.fetch(sql, params)
        return results[0] if results else None

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Execute a query and return the first column of the first row."""
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        self._in_use = True
        try:
            _, rows = await protocol.query_sql(db_id, sql, params)
        except (DqliteConnectionError, ProtocolError):
            self._invalidate()
            raise
        except OperationalError as e:
            if e.code in _LEADER_ERROR_CODES:
                self._invalidate()
            raise
        except BaseException:
            self._invalidate()
            raise
        finally:
            self._in_use = False
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

        self._in_transaction = True
        self._tx_owner = asyncio.current_task()
        try:
            await self.execute("BEGIN")
        except BaseException:
            self._tx_owner = None
            self._in_transaction = False
            raise

        try:
            yield
            await self.execute("COMMIT")
        except BaseException:
            # Swallow rollback failure; original exception is more important
            with contextlib.suppress(BaseException):
                await self.execute("ROLLBACK")
            raise
        finally:
            self._tx_owner = None
            self._in_transaction = False
