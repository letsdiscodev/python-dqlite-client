"""A single wire session to one dqlite node."""

import asyncio
import contextlib
import logging
import os
import warnings
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence, Sized
from types import TracebackType
from typing import Any, Final, NoReturn, Self

from dqliteclient._dial import DialFunc, open_connection
from dqliteclient._validate import (
    CLOSE_TIMEOUT_FLOOR,
    CLOSE_TIMEOUT_FLOOR_RATIONALE,
    DEFAULT_CLOSE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    parse_address,
    validate_timeout,
)
from dqliteclient.exceptions import (
    AmbiguousCommitError,
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.protocol import DqliteProtocol, validate_positive_int_or_none
from dqliteclient.sql import (
    is_keyword_boundary,
    leading_keyword,
    split_statements,
    strip_leading_comments,
)
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    LEADER_ERROR_CODES,
    LEADER_LOST_DB_LOOKUP_SUBSTRING,
    NO_TRANSACTION_MESSAGE_SUBSTRINGS,
    SQLITE_BUSY,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
    SQLITE_NOTFOUND,
    WIRE_DECODE_FAILED_PREFIX,
    EncodeError,
    primary_sqlite_code,
    sanitize_for_log,
)
from dqlitewire.constants import TX_AUTO_ROLLBACK_PRIMARY_CODES

__all__ = ["DqliteConnection"]

logger = logging.getLogger(__name__)

_RAFT_BUSY_FRAGMENT: Final[str] = "checkpoint in progress"
_LEADERSHIP_LOST_CODES: Final[frozenset[int]] = frozenset(
    {SQLITE_IOERR_LEADERSHIP_LOST, SQLITE_IOERR_LEADERSHIP_LOST_LEGACY}
)


def is_no_transaction_error(exc: BaseException) -> bool:
    """True for the server's "no transaction is active" reply to COMMIT / ROLLBACK."""
    if not isinstance(exc, OperationalError) or primary_sqlite_code(exc.code) != 1:
        return False
    text = (exc.raw_message or exc.message).lower()
    return any(s in text for s in NO_TRANSACTION_MESSAGE_SUBSTRINGS)


def _is_leader_change(exc: OperationalError) -> bool:
    if exc.code in LEADER_ERROR_CODES:
        return True
    text = (exc.raw_message or exc.message or "").lower()
    return exc.code == SQLITE_NOTFOUND and text.startswith(LEADER_LOST_DB_LOOKUP_SUBSTRING)


class _State:
    """Mutable cells the ResourceWarning finalizer reads after the connection is gone."""

    __slots__ = ("address", "connected", "closed", "pid")

    def __init__(self, address: str) -> None:
        self.address = address
        self.connected = False
        self.closed = False
        self.pid = os.getpid()


def _warn_if_unclosed(state: _State) -> None:
    if state.closed or not state.connected or os.getpid() != state.pid:
        return
    warnings.warn(
        f"DqliteConnection(address={sanitize_for_log(state.address)!r}) was garbage-collected "
        "without await close(); call close() to release the socket promptly.",
        ResourceWarning,
        stacklevel=2,
    )


class DqliteConnection:
    """One wire session to a dqlite node. Not thread-safe; one operation at a time.

    Binds to the event loop it first runs on. A lost session (transport error, leader
    change, cancellation mid round-trip) closes the connection for good.
    """

    def __init__(
        self,
        address: str,
        *,
        database: str = "default",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        dial_func: DialFunc | None = None,
        max_message_size: int | None = None,
    ) -> None:
        validate_timeout(timeout)
        validate_timeout(
            close_timeout,
            name="close_timeout",
            min_value=CLOSE_TIMEOUT_FLOOR,
            min_value_rationale=CLOSE_TIMEOUT_FLOOR_RATIONALE,
        )
        if dial_timeout is not None:
            validate_timeout(dial_timeout, name="dial_timeout")
        if attempt_timeout is not None:
            validate_timeout(attempt_timeout, name="attempt_timeout")
        parse_address(address)
        self._address = address
        self._database = database
        self._timeout = timeout
        self._dial_timeout = dial_timeout if dial_timeout is not None else timeout
        self._attempt_timeout = attempt_timeout if attempt_timeout is not None else timeout
        self._close_timeout = close_timeout
        self._dial_func = dial_func
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._max_message_size = max_message_size
        self._trust_server_heartbeat = trust_server_heartbeat
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False
        self._busy = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state = _State(address)
        self._finalizer = weakref.finalize(self, _warn_if_unclosed, self._state)

    # -- state -------------------------------------------------------------------

    @property
    def address(self) -> str:
        return self._address

    @property
    def is_connected(self) -> bool:
        """True while the wire session is open and its transport is alive."""
        return self._protocol is not None and self._protocol.is_alive

    @property
    def closed(self) -> bool:
        """True once :meth:`close` or :meth:`terminate` ran."""
        return self._state.closed

    @property
    def in_transaction(self) -> bool:
        """Conservative local view of whether a transaction is open; see docs/architecture.md."""
        return self._in_transaction

    def __repr__(self) -> str:
        if self.closed:
            status = "closed"
        elif self._protocol is not None:
            status = "connected"
        else:
            status = "disconnected"
        return (
            f"<DqliteConnection address={sanitize_for_log(self._address)!r} "
            f"database={self._database!r} {status} at 0x{id(self):x}>"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"cannot pickle {type(self).__name__!r}: it owns a live socket")

    # -- guards ------------------------------------------------------------------

    def _check_usable(self) -> None:
        if os.getpid() != self._state.pid:
            raise InterfaceError(
                f"Connection used after fork; reconstruct it in the child process "
                f"(created in pid {self._state.pid}, current pid {os.getpid()})"
            )
        if self._state.closed:
            raise InterfaceError(f"Connection is closed (id={id(self)})")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise InterfaceError(
                "DqliteConnection is bound to a different event loop; do not share "
                "connections across loops or threads"
            )

    @contextlib.contextmanager
    def _operation(self) -> Any:
        self._check_usable()
        if self._busy:
            raise InterfaceError(
                "Cannot perform operation: another operation is in progress on this "
                "connection. DqliteConnection does not support concurrent access; "
                "use one connection per task or a ConnectionPool."
            )
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    @staticmethod
    def _validate_params(params: object) -> None:
        if params is None:
            return
        if isinstance(params, str | bytes | bytearray | memoryview):
            raise DataError(
                f"params must be a sequence of values, not {type(params).__name__}; "
                "wrap a single value in a tuple"
            )
        if isinstance(params, Mapping):
            raise DataError("params must be a positional sequence, not a mapping")
        if isinstance(params, set | frozenset):
            raise DataError("params must be an ordered sequence, not a set")
        if not isinstance(params, Sized):
            raise DataError(
                f"params must be a sized sequence, not {type(params).__name__}; "
                "single-shot iterators are rejected because a retry could not replay them"
            )

    # -- lifecycle ---------------------------------------------------------------

    async def connect(self) -> None:
        """Dial, handshake and open the database. Idempotent while connected."""
        with self._operation():
            if self._protocol is not None:
                return
            address = sanitize_for_log(self._address)
            protocol: DqliteProtocol | None = None
            phase = "dial"
            try:
                async with asyncio.timeout(self._attempt_timeout):
                    async with asyncio.timeout(self._dial_timeout):
                        reader, writer = await open_connection(
                            self._address, dial_func=self._dial_func
                        )
                    phase = "handshake"
                    protocol = DqliteProtocol(
                        reader,
                        writer,
                        timeout=self._timeout,
                        max_total_rows=self._max_total_rows,
                        max_continuation_frames=self._max_continuation_frames,
                        trust_server_heartbeat=self._trust_server_heartbeat,
                        address=self._address,
                        max_message_size=self._max_message_size,
                    )
                    await protocol.handshake()
                    db_id = await protocol.open_database(self._database)
            except BaseException as exc:
                if protocol is not None:
                    protocol.close()
                if isinstance(exc, TimeoutError):
                    if phase == "dial":
                        raise DqliteConnectionError(
                            f"Connection to {address} timed out "
                            f"(dial_timeout={self._dial_timeout}s)"
                        ) from exc
                    raise DqliteConnectionError(
                        f"Handshake with {address} timed out "
                        f"(attempt_timeout={self._attempt_timeout}s)"
                    ) from exc
                if isinstance(exc, OSError):
                    raise DqliteConnectionError(f"Failed to connect to {address}: {exc}") from exc
                if isinstance(exc, OperationalError) and _is_leader_change(exc):
                    raise DqliteConnectionError(
                        f"Node {address} is no longer leader: {exc.message}",
                        code=exc.code,
                        raw_message=exc.raw_message,
                    ) from exc
                if isinstance(exc, ProtocolError):
                    raise DqliteConnectionError(
                        f"{WIRE_DECODE_FAILED_PREFIX} during handshake to {address}: {exc}",
                        raw_message=exc.raw_message or str(exc),
                    ) from exc
                raise
            self._protocol = protocol
            self._db_id = db_id
            self._state.connected = True
            logger.debug("connected to %s (database=%r, db_id=%d)", address, self._database, db_id)

    def _invalidate(self) -> None:
        protocol, self._protocol = self._protocol, None
        self._db_id = None
        self._in_transaction = False
        if protocol is not None:
            with contextlib.suppress(Exception):
                protocol.close()

    def terminate(self) -> None:
        """Drop the transport synchronously; idempotent, never raises."""
        self._state.closed = True
        self._finalizer.detach()
        self._invalidate()

    async def close(self) -> None:
        """Close the connection, waiting up to ``close_timeout`` for the transport. Idempotent."""
        if self._state.closed:
            return
        self._state.closed = True
        self._finalizer.detach()
        protocol, self._protocol = self._protocol, None
        self._db_id = None
        self._in_transaction = False
        if protocol is None or os.getpid() != self._state.pid:
            return
        protocol.close()
        with contextlib.suppress(OSError, TimeoutError):
            async with asyncio.timeout(self._close_timeout):
                await protocol.wait_closed()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            await self.close()
        except Exception:
            if exc_val is None:
                raise
            logger.debug(
                "close failed in __aexit__ while %s was propagating",
                type(exc_val).__name__,
                exc_info=True,
            )

    # -- RPC plumbing --------------------------------------------------------------

    async def _run[T](self, fn: Callable[[DqliteProtocol, int], Awaitable[T]]) -> T:
        with self._operation():
            if self._protocol is None or self._db_id is None:
                raise DqliteConnectionError("Not connected")
            try:
                return await fn(self._protocol, self._db_id)
            except EncodeError as exc:
                raise DataError(f"wire encode failed: {exc}") from exc
            except (DqliteConnectionError, ProtocolError):
                self._invalidate()
                raise
            except OperationalError as exc:
                self._on_operational_error(exc)
                raise
            except BaseException:
                # Cancelled or interrupted mid round-trip: the wire position is unknown.
                self._invalidate()
                raise

    def _on_operational_error(self, exc: OperationalError) -> None:
        if _is_leader_change(exc):
            self._invalidate()
            return
        primary = primary_sqlite_code(exc.code)
        if primary in TX_AUTO_ROLLBACK_PRIMARY_CODES or is_no_transaction_error(exc):
            self._in_transaction = False
        elif primary == SQLITE_BUSY and _RAFT_BUSY_FRAGMENT in (exc.raw_message or "").lower():
            # A raft checkpoint discarded the in-flight transaction; the session is unusable.
            self._invalidate()
            raise DqliteConnectionError(
                f"raft-checkpoint reset the in-flight transaction: {exc}",
                code=exc.code,
                raw_message=exc.raw_message,
            ) from exc

    def _track_transaction(self, sql: str) -> None:
        for statement in split_statements(sql):
            keyword = leading_keyword(statement)
            if keyword in ("BEGIN", "SAVEPOINT"):
                self._in_transaction = True
            elif (
                keyword in ("COMMIT", "END")
                or keyword == "ROLLBACK"
                and not _is_rollback_to(statement)
            ):
                self._in_transaction = False

    # -- SQL -------------------------------------------------------------------------

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a statement; return ``(last_insert_id, rows_affected)``."""
        self._validate_params(params)
        result = await self._run(lambda p, db: p.exec_sql(db, sql, params))
        self._track_transaction(sql)
        return result

    async def query_raw(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Run a query; return ``(column_names, rows)``."""
        self._validate_params(params)
        return await self._run(lambda p, db: p.query_sql(db, sql, params))

    async def query_raw_typed(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """Run a query; return ``(column_names, column_types, row_types, rows)`` with wire
        ``ValueType`` codes per column (first row) and per row."""
        self._validate_params(params)
        return await self._run(lambda p, db: p.query_sql_typed(db, sql, params))

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        columns, rows = await self.query_raw(sql, params)
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        _, rows = await self.query_raw(sql, params)
        return rows

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        """First row as a dict, or ``None``. The server still sends every row; add ``LIMIT``."""
        rows = await self.fetch(sql, params)
        return rows[0] if rows else None

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """First column of the first row, or ``None``."""
        _, rows = await self.query_raw(sql, params)
        return rows[0][0] if rows and rows[0] else None

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """``BEGIN`` on entry, ``COMMIT`` on clean exit, ``ROLLBACK`` if the body raises.

        Losing leadership during ``COMMIT`` raises :class:`AmbiguousCommitError`: the
        write may or may not have been applied.
        """
        if self._in_transaction:
            raise InterfaceError("Nested transactions are not supported; use SAVEPOINT directly")
        await self.execute("BEGIN")
        try:
            yield
        except BaseException:
            await self._rollback_after()
            raise
        try:
            await self.execute("COMMIT")
        except OperationalError as exc:
            if exc.code in _LEADERSHIP_LOST_CODES:
                raise AmbiguousCommitError(
                    f"leadership lost during COMMIT; the transaction may or may not have "
                    f"been applied: {exc.message}",
                    exc.code,
                    raw_message=exc.raw_message,
                ) from exc
            raise

    async def _rollback_after(self) -> None:
        where = f"(address={sanitize_for_log(self._address)}, id={id(self)})"
        try:
            await self.execute("ROLLBACK")
        except asyncio.CancelledError:
            logger.debug("transaction rollback was cancelled %s", where)
            self._invalidate()
            raise
        except OperationalError as exc:
            if not is_no_transaction_error(exc):
                logger.debug("transaction rollback failed %s", where, exc_info=True)
                self._invalidate()
        except Exception:
            logger.debug("transaction rollback failed %s", where, exc_info=True)
            self._invalidate()


def _is_rollback_to(statement: str) -> bool:
    rest = strip_leading_comments(statement)[len("ROLLBACK") :].lstrip()
    upper = rest.upper()
    if upper.startswith("TRANSACTION") and is_keyword_boundary(upper, len("TRANSACTION")):
        upper = upper[len("TRANSACTION") :].lstrip()
    return upper.startswith("TO") and is_keyword_boundary(upper, 2)
