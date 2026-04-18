"""High-level connection interface for dqlite."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dqliteclient.exceptions import (
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.exceptions import EncodeError as _WireEncodeError

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

    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port in address {address!r}: {port} is not in range 1-65535")
    if not host:
        raise ValueError(f"Invalid address format: empty hostname in {address!r}")
    if host.count(":") > 1 and not address.startswith("["):
        raise ValueError(
            f"IPv6 addresses must be bracketed: use '[{host}]:{port}' instead of {address!r}"
        )

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
        import math

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"timeout must be a positive finite number, got {timeout}")
        self._address = address
        self._database = database
        self._timeout = timeout
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False
        self._in_use = False
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tx_owner: asyncio.Task[Any] | None = None
        self._pool_released = False
        self._invalidation_cause: BaseException | None = None

    @property
    def address(self) -> str:
        """Get the connection address."""
        return self._address

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._protocol is not None

    def __repr__(self) -> str:
        state = "connected" if self._protocol is not None else "disconnected"
        return (
            f"<DqliteConnection address={self._address!r} database={self._database!r} {state}>"
        )

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
            except OperationalError as e:
                self._protocol.close()
                self._protocol = None
                if e.code in _LEADER_ERROR_CODES:
                    # Leader-change errors during OPEN are transport-level
                    # problems — the caller needs to reconnect elsewhere, not
                    # treat this as a SQL error.
                    raise DqliteConnectionError(
                        f"Node {self._address} is no longer leader: {e.message}"
                    ) from e
                raise
            except BaseException:
                self._protocol.close()
                self._protocol = None
                raise
        finally:
            self._in_use = False

    async def close(self) -> None:
        """Close the connection.

        Idempotent: safe to call on an already-closed or pool-released
        connection. Null ``_protocol`` before awaiting ``wait_closed`` so
        a concurrent second close cannot re-enter the socket-close path.
        """
        # Pool-released or already-closed: nothing to do.
        if self._pool_released or self._protocol is None:
            return
        self._check_in_use()
        protocol = self._protocol
        self._protocol = None
        self._db_id = None
        protocol.close()
        await protocol.wait_closed()

    async def __aenter__(self) -> "DqliteConnection":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _ensure_connected(self) -> tuple[DqliteProtocol, int]:
        """Ensure we're connected and return protocol and db_id."""
        if self._protocol is None or self._db_id is None:
            raise DqliteConnectionError("Not connected") from self._invalidation_cause
        return self._protocol, self._db_id

    def _check_in_use(self) -> None:
        """Raise on misuse: wrong event loop, concurrent access, or use after pool release."""
        if self._pool_released:
            raise InterfaceError(
                "This connection has been returned to the pool and can no longer "
                "be used directly. Acquire a new connection from the pool."
            )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            raise InterfaceError(
                "DqliteConnection must be used from within an async context."
            ) from None
        if self._bound_loop is None:
            # Lazily bind on first use so the guard is always active, even
            # for bare-instantiation / mocked-protocol patterns that skip
            # connect().
            self._bound_loop = current_loop
        elif current_loop is not self._bound_loop:
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

    def _invalidate(self, cause: BaseException | None = None) -> None:
        """Mark the connection as broken after an unrecoverable error.

        If ``cause`` is provided, it is remembered so a later caller that
        hits "Not connected" can chain it as ``__cause__`` for diagnostics.
        """
        if self._protocol is not None:
            # Connection may already be broken; suppress close errors
            with contextlib.suppress(Exception):
                self._protocol.close()
        self._protocol = None
        self._db_id = None
        if cause is not None:
            self._invalidation_cause = cause

    @staticmethod
    def _validate_params(params: Sequence[Any] | None) -> None:
        """Reject bare str/bytes params to catch the ``execute("?", "x")`` footgun.

        ``str`` and ``bytes`` are both ``Sequence[Any]``, so they type-check
        but would silently split into N single-character parameters.
        """
        if isinstance(params, str | bytes):
            raise TypeError("params must be a list or tuple, not str/bytes; did you mean [value]?")

    async def _run_protocol[T](self, fn: Callable[[DqliteProtocol, int], Awaitable[T]]) -> T:
        """Run a protocol operation with standard error handling.

        Handles connection guards (_check_in_use, _ensure_connected, _in_use),
        invalidates the connection on fatal errors, and resets _in_use in all cases.
        """
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        self._in_use = True
        try:
            return await fn(protocol, db_id)
        except _WireEncodeError as e:
            # Client-side parameter-encoding error. The wire bytes were
            # never written, so the connection is still usable — convert
            # into the client-level DataError and let it propagate.
            raise DataError(str(e)) from e
        except (DqliteConnectionError, ProtocolError) as e:
            self._invalidate(e)
            raise
        except OperationalError as e:
            if e.code in _LEADER_ERROR_CODES:
                self._invalidate(e)
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit) as e:
            # Interrupted mid-operation; we don't know how much of the
            # request/response round-trip completed, so the wire state is
            # unsafe to reuse. Invalidate and re-raise.
            self._invalidate(e)
            raise
        finally:
            self._in_use = False

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement.

        Returns (last_insert_id, rows_affected).
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.exec_sql(db, sql, params))

    async def query_raw(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query and return raw (column_names, rows).

        Unlike fetch() which returns dicts, this returns the raw tuple
        of (column_names, rows) from the wire protocol. Intended for
        DBAPI cursor implementations that need column names separately.
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))

    async def query_raw_typed(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[Any]]]:
        """Execute a query and return (column_names, column_types, rows).

        ``column_types`` are per-column wire ``ValueType`` integer tags
        from the first response frame — suitable for populating DBAPI
        ``cursor.description[i][1]`` (``type_code``).
        """
        return await self._run_protocol(lambda p, db: p.query_sql_typed(db, sql, params))

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        self._validate_params(params)
        columns, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        """Execute a query and return results as list of lists."""
        self._validate_params(params)
        _, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        return rows

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute a query and return the first result.

        Note: dqlite returns all matching rows over the wire. For large
        result sets, add ``LIMIT 1`` to your query to avoid excessive
        memory usage.
        """
        results = await self.fetch(sql, params)
        return results[0] if results else None

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Execute a query and return the first column of the first row."""
        self._validate_params(params)
        _, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        if rows and rows[0]:
            return rows[0][0]
        return None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for transactions."""
        if self._in_transaction:
            raise InterfaceError("Nested transactions are not supported; use SAVEPOINT directly")

        self._in_transaction = True
        self._tx_owner = asyncio.current_task()
        try:
            await self.execute("BEGIN")
        except BaseException:
            self._tx_owner = None
            self._in_transaction = False
            raise

        commit_attempted = False
        try:
            yield
            commit_attempted = True
            await self.execute("COMMIT")
        except BaseException:
            if commit_attempted:
                # COMMIT was sent but failed. Server-side state is ambiguous
                # (maybe committed, maybe still open, maybe rolled back). We
                # cannot safely reuse this connection — invalidate so the
                # pool discards it instead of recycling an unknown state.
                self._invalidate()
            else:
                # Body raised before COMMIT; try to roll back. Swallow
                # rollback errors; the original exception is more important.
                with contextlib.suppress(BaseException):
                    await self.execute("ROLLBACK")
            raise
        finally:
            self._tx_owner = None
            self._in_transaction = False
