"""High-level connection interface for dqlite."""

import asyncio
import contextlib
import ipaddress
import logging
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from dqliteclient.exceptions import (
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.protocol import (
    DqliteProtocol,
    _validate_positive_int_or_none,
)
from dqlitewire import LEADER_ERROR_CODES as _LEADER_ERROR_CODES
from dqlitewire import primary_sqlite_code as _primary_sqlite_code
from dqlitewire.exceptions import EncodeError as _WireEncodeError

__all__ = ["DqliteConnection"]

logger = logging.getLogger(__name__)

# Bare ``BEGIN`` opens an implicit ``DEFERRED`` transaction per SQLite's
# grammar default. dqlite's Raft FSM serializes transactions regardless
# of the qualifier, so DEFERRED / IMMEDIATE / EXCLUSIVE collapse to the
# same SERIALIZABLE semantics on dqlite — there is no benefit to
# emitting an explicit qualifier and doing so would diverge from the C
# and Go peer clients which also emit a bare ``BEGIN``. The constant
# pins the literal so a refactor can't silently upgrade to ``BEGIN
# IMMEDIATE`` without showing up in review.
_TRANSACTION_BEGIN_SQL = "BEGIN"

# ``COMMIT`` and ``ROLLBACK`` literals are pinned for the same reason as
# ``_TRANSACTION_BEGIN_SQL``: a refactor must not silently switch to a
# vendor-qualified form (``COMMIT TRANSACTION``) or a savepoint variant.
# Consistency between ``transaction()`` and the pool's reset-on-return
# path is a correctness invariant — if they diverge, a connection could
# be returned to the pool with a still-open server-side transaction.
_TRANSACTION_COMMIT_SQL = "COMMIT"
_TRANSACTION_ROLLBACK_SQL = "ROLLBACK"

# Primary SQLite error codes whose semantics imply the server-side
# SQLite transaction was automatically rolled back. Upstream
# ``leader.c`` polls ``sqlite3_txn_state`` after each exec and clears
# ``active_leader`` when the engine reports ``SQLITE_TXN_NONE``, so
# the cluster-side tx is gone for any of these primary codes:
#
#   * SQLITE_ABORT (4) — operation aborted, e.g. via sqlite3_interrupt.
#   * SQLITE_INTERRUPT (9) — query interrupted via INTERRUPT.
#   * SQLITE_IOERR (10) — and most extended IOERR variants (the leader-
#     change variants are caught earlier as ``_LEADER_ERROR_CODES`` and
#     trigger a full ``_invalidate`` instead).
#   * SQLITE_CORRUPT (11).
#   * SQLITE_FULL (13).
#
# When ``_run_protocol`` sees one of these on a non-leader-class
# OperationalError, the underlying connection is still healthy but
# ``_in_transaction`` / ``_tx_owner`` must be cleared — otherwise the
# Python side reports True for ``in_transaction`` while the server
# reports tx-none, and the next user statement implicitly auto-begins
# a fresh transaction whose boundary the user code does not know about.
_TX_AUTO_ROLLBACK_PRIMARY_CODES = frozenset({4, 9, 10, 11, 13})


def _is_no_tx_rollback_error(exc: BaseException) -> bool:
    """True if ``exc`` is the deterministic "no transaction is active"
    reply from the server during a ROLLBACK.

    Recognised by SQLite primary code 1 (``SQLITE_ERROR``) plus the
    "no transaction is active" or "cannot rollback" wording. Both
    conditions must hold so a disk-full / constraint / IO error whose
    message happens to include the magic substring is not silently
    treated as benign.

    Used by the ``transaction()`` context manager and by the pool's
    ``_reset_connection`` to distinguish "server already auto-rolled
    back; preserve the slot" from a real ROLLBACK failure that
    requires invalidation. Centralising the check avoids drift if the
    SQLite-error wording ever changes.
    """
    if not isinstance(exc, OperationalError):
        return False
    code = getattr(exc, "code", None)
    if code is None or _primary_sqlite_code(code) != 1:  # SQLITE_ERROR
        return False
    msg = str(exc).lower()
    return "no transaction is active" in msg or "cannot rollback" in msg


# RFC 1035 hostname labels are ASCII letters, digits, and hyphen. We
# accept a dotted sequence of labels up to 253 chars total. Single
# labels (e.g. "localhost") are also accepted.
_HOSTNAME_LABEL_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)"
    r"(?:\.(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?))*$"
)


def _canonicalize_host(host: str, address: str) -> str:
    """Validate and canonicalize a host portion of an address.

    Accepts IPv4 literals, IPv6 literals (already unwrapped from
    brackets), and ASCII hostnames. Returns the canonical form:
    ``ipaddress.ip_address(h)`` for IP literals, lowercase for
    hostnames. Rejects credentials-like '@', whitespace/CRLF, and
    non-ASCII (IDN) hosts so a server-controlled redirect target
    cannot smuggle log-injection or DNS-rebinding vectors past the
    parser.
    """
    if not host:
        raise ValueError(f"Invalid address format: empty hostname in {address!r}")
    # Try IP literal first — IPv6 shorthand (``::1``) must canonicalize
    # so allowlists see one form regardless of how the peer wrote it.
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    # ASCII-only: reject IDN outright. dqlite's wire does not round-
    # trip punycode reliably, and non-ASCII hostnames are a common
    # homograph-attack vector.
    try:
        host.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(
            f"Invalid host in address {address!r}: non-ASCII hostnames are not supported"
        ) from e
    if not _HOSTNAME_LABEL_RE.match(host):
        raise ValueError(
            f"Invalid host in address {address!r}: {host!r} is not a valid hostname or IP literal"
        )
    return host.lower()


def _validate_timeout(value: float, *, name: str = "timeout") -> float:
    """Validate a user-supplied timeout: positive, finite, not ``bool``.

    ``isinstance(True, int)`` is True and ``math.isfinite(True)`` is
    True, so a bare ``<= 0`` check lets ``timeout=True`` through as
    ``1.0``. Reject ``bool`` explicitly. Also reject ``inf`` / ``nan``
    so a misconfigured value fails at the caller's entry point rather
    than much later when it reaches ``asyncio.wait_for``.

    Shared across ``DqliteConnection``, ``ClusterClient``, and
    ``ConnectionPool`` so every entry point enforces the same contract.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r} (bool)")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return float(value)


def _parse_address(address: str) -> tuple[str, int]:
    """Parse a host:port address string, handling IPv6 brackets.

    Returns ``(canonical_host, port)``. IP literals are returned in
    ``ipaddress.ip_address``'s canonical form; hostnames are
    lowercased. Invalid hosts (credentials-like '@', whitespace/CRLF,
    non-ASCII, empty) raise ``ValueError``.
    """
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
        # Diagnose unbracketed IPv6 BEFORE attempting to parse the
        # port: ``"::1:abc"`` splits to host="::1", port_str="abc",
        # and the ``int()`` failure would surface as "invalid port"
        # when the real error is the missing brackets. Require the
        # host to contain ``:`` and NOT contain ``@`` — the latter
        # indicates a credentials-smuggle shape (rejected by
        # ``_canonicalize_host`` below with a more specific message).
        if ":" in host and "@" not in host:
            raise ValueError(
                f"IPv6 addresses must be bracketed: got {address!r}, expected '[host]:port'"
            )

    # Strict port parse: stdlib ``int()`` accepts whitespace, unary
    # ``+``, PEP 515 underscores, leading zeros, and Unicode digits.
    # An ``allowlist_policy`` comparing against a configured address
    # set would then fail to match when the peer redirects to a
    # non-canonical form (``"host: 9000 "``, ``"host:+9000"``,
    # ``"host:9_000"``) even though it's semantically the same port.
    # Restrict to plain ASCII digits (with an optional leading ``-``
    # so negative ports surface the "not in range" diagnostic rather
    # than a confusing "not a number" one).
    if port_str.startswith("-") and port_str[1:].isascii() and port_str[1:].isdigit():
        port = int(port_str)  # negative — fails the range check below
    elif port_str.isascii() and port_str.isdigit():
        port = int(port_str)
    else:
        raise ValueError(f"Invalid port in address {address!r}: {port_str!r} is not a number")

    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port in address {address!r}: {port} is not in range 1-65535")

    host = _canonicalize_host(host, address)
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
        max_total_rows: int | None = 10_000_000,
        max_continuation_frames: int | None = 100_000,
        trust_server_heartbeat: bool = False,
        close_timeout: float = 0.5,
    ) -> None:
        """Initialize connection (does not connect yet).

        Args:
            address: Node address in "host:port" format
            database: Database name to open
            timeout: Connection timeout in seconds
            max_total_rows: Cumulative row cap across continuation
                frames for a single query. Prevents a slow-drip server
                from keeping the client alive indefinitely within the
                per-operation deadline. Set to ``None`` to disable.
            max_continuation_frames: Maximum number of continuation
                frames in a single query result. Caps the per-query
                Python-side decode work a hostile server can inflict
                by sending many 1-row frames. Set to
                ``None`` to disable.
            trust_server_heartbeat: When True, widen the per-read
                deadline to the server-advertised heartbeat (subject
                to a 300 s hard cap). When False (default), ``timeout``
                is authoritative — the server value cannot amplify it.
            close_timeout: Budget in seconds for the transport-drain
                half of ``close()``. After ``writer.close()`` the
                local side of the socket is gone; ``wait_closed``
                is best-effort cleanup. An unresponsive peer must not
                stall ``engine.dispose()`` or SIGTERM shutdown, so
                the drain is bounded by this value.
        """
        _validate_timeout(timeout)
        _validate_timeout(close_timeout, name="close_timeout")
        # Parse at construction so a misconfigured address (typoed DSN,
        # invalid port, unbracketed IPv6) raises ValueError at the
        # operator's config-load site rather than inside connect(),
        # where SA's is_disconnect substring scan would mis-classify
        # it as a retryable transport failure and loop.
        self._host, self._port = _parse_address(address)
        self._address = address
        self._database = database
        self._timeout = timeout
        self._close_timeout = close_timeout
        self._max_total_rows = _validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = _validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._trust_server_heartbeat = trust_server_heartbeat
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False
        self._in_use = False
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tx_owner: asyncio.Task[Any] | None = None
        self._pool_released = False
        # Cause recorded by ``_invalidate(cause=...)``; only meaningful
        # while ``_protocol is None``. ``connect()`` clears it on a
        # successful re-handshake so later "Not connected" errors don't
        # chain to an ancient unrelated failure.
        self._invalidation_cause: BaseException | None = None
        # Tracks the bounded ``wait_closed`` drain scheduled by
        # ``_invalidate`` so a subsequent ``close()`` can await it and
        # keep the reader task from outliving the connection.
        self._pending_drain: asyncio.Task[None] | None = None

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
        return f"<DqliteConnection address={self._address!r} database={self._database!r} {state}>"

    @property
    def in_transaction(self) -> bool:
        """Check if a transaction is active."""
        return self._in_transaction

    async def connect(self) -> None:
        """Establish connection to the database."""
        self._check_in_use()
        if self._protocol is not None:
            return

        # If a prior ``_invalidate`` scheduled a bounded drain task,
        # retire it here before the slot gets reused. Leaving the
        # previous task in place would let a second invalidate at
        # line ~483 overwrite the slot without cancelling or awaiting
        # it, breaking the "strong ref so close() can await it"
        # discipline documented on that assignment.
        pending = self._pending_drain
        if pending is not None:
            if not pending.done():
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            self._pending_drain = None

        self._bound_loop = asyncio.get_running_loop()
        self._in_use = True
        try:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self._host, self._port),
                    timeout=self._timeout,
                )
            except TimeoutError as e:
                raise DqliteConnectionError(f"Connection to {self._address} timed out") from e
            except OSError as e:
                raise DqliteConnectionError(f"Failed to connect to {self._address}: {e}") from e

            try:
                self._protocol = DqliteProtocol(
                    reader,
                    writer,
                    timeout=self._timeout,
                    max_total_rows=self._max_total_rows,
                    max_continuation_frames=self._max_continuation_frames,
                    trust_server_heartbeat=self._trust_server_heartbeat,
                    address=self._address,
                )
            except BaseException:
                # Protocol construction is currently limited to argument
                # validation, which ``DqliteConnection.__init__`` already
                # enforces — but if it ever raises (now or through future
                # refactors), ``_abort_protocol`` is a no-op until
                # ``self._protocol`` is assigned, so reader/writer would
                # be leaked. Close the transport defensively.
                writer.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(writer.wait_closed(), timeout=self._close_timeout)
                raise

            try:
                await self._protocol.handshake()
                logger.debug(
                    "connect: handshake ok address=%s client_id=%d",
                    self._address,
                    self._protocol._client_id,
                )
                self._db_id = await self._protocol.open_database(self._database)
                logger.debug(
                    "connect: db opened address=%s db_id=%d database=%r",
                    self._address,
                    self._db_id,
                    self._database,
                )
                # Clear any stale cause recorded by a prior ``_invalidate``.
                # The field is only meaningful while ``_protocol is None``;
                # a successful reconnect supersedes it. Without this,
                # a later silent invalidation (``_invalidate()`` with no
                # cause) would produce "Not connected" errors whose
                # ``__cause__`` chain points back at an unrelated historical
                # failure — misleading operators reading logs.
                self._invalidation_cause = None
            except OperationalError as e:
                await self._abort_protocol()
                if e.code in _LEADER_ERROR_CODES:
                    # Leader-change errors during OPEN are transport-level
                    # problems — the caller needs to reconnect elsewhere, not
                    # treat this as a SQL error.
                    raise DqliteConnectionError(
                        f"Node {self._address} is no longer leader: {e.message}"
                    ) from e
                raise
            except BaseException:
                await self._abort_protocol()
                raise
        finally:
            self._in_use = False
            # On a never-connected failure path — ``connect()`` raised
            # before ``_protocol`` was published — the loop binding
            # recorded above at entry is historical noise. A retry
            # from a different event loop (legitimate in unit-test
            # teardown or ``asyncio.run()`` re-entry) would otherwise
            # fail ``_check_in_use`` with "bound to a different event
            # loop" even though no successful connection ever existed.
            # Drop the bind on the failure path so the instance stays
            # usable.
            if self._protocol is None:
                self._bound_loop = None

    async def close(self) -> None:
        """Close the connection.

        Idempotent: safe to call on an already-closed or pool-released
        connection. Null ``_protocol`` before awaiting ``wait_closed``
        so a concurrent second close cannot re-enter the socket-close
        path.

        The transport drain (``wait_closed``) is bounded by
        ``close_timeout``. ``close()`` is the hot path on every pool
        release and on ``engine.dispose()`` / SIGTERM shutdown; an
        unresponsive peer must not be able to stall shutdown by
        refusing to acknowledge a FIN. The local side of the socket
        is already closed after ``writer.close()`` — the remaining
        wait is best-effort cleanup, not correctness-critical.
        """
        # Pool-released connections are never in_use for close(); their
        # close path has already run under pool ownership.
        if self._pool_released:
            return
        # Run the in-use guard BEFORE the ``_protocol is None``
        # early-return so a concurrent ``connect()`` racing with
        # ``close()`` surfaces as ``InterfaceError`` instead of a silent
        # no-op. Without this, close() returning while connect() is
        # suspended in ``asyncio.open_connection`` would leak the
        # eventual socket — connect() publishes _protocol only on
        # success, so at the race moment _protocol is None and
        # close() would silently return.
        self._check_in_use()
        # ``_invalidate`` may have scheduled a bounded drain task on
        # the writer it just closed. Await it so the reader task exits
        # cleanly; otherwise Python logs "Task was destroyed but it is
        # pending" at interpreter shutdown. Drop the slot either way
        # so a subsequent close/connect cycle starts fresh — clearing
        # only inside the ``_protocol is None`` branch left a done-task
        # reference on the happy path indefinitely, inconsistent with
        # ``connect()``'s own ``_pending_drain = None`` symmetry.
        pending = self._pending_drain
        self._pending_drain = None
        if pending is not None and not pending.done():  # pragma: no cover
            # Defensive: the pending bounded-drain task is set only
            # by ``_invalidate`` and is normally already-done by
            # the time ``close()`` runs (the drain is bounded by
            # ``close_timeout`` and the close path is the second
            # caller). Reaching here requires the rare race where
            # ``_invalidate`` fires between the snapshot read and
            # the second caller's resumption — verified by code
            # review, not coverage.
            with contextlib.suppress(Exception):
                await pending
        # Mirror ``_invalidate``'s atomic clear of the transaction
        # bookkeeping. Without this, a raw ``BEGIN`` followed by an
        # explicit ``close()`` and a reconnect on the same instance
        # leaves ``_in_transaction=True`` (and possibly a stale
        # ``_tx_owner``) — the next caller's ``in_transaction`` lies
        # about server-side state, and ``transaction()`` trips the
        # "Nested transactions are not supported" guard with no real
        # nesting happening. Place the clear before the
        # ``_protocol is None`` early-return so an already-closed
        # connection that still carries stale flags (e.g. from a prior
        # raw BEGIN whose close was concurrent with this one) gets
        # them scrubbed too. Pool-released connections take the
        # early-return at the top of the method, so the pool's
        # ``_reset_connection`` remains the canonical clear for that
        # path; this clear covers direct ``DqliteConnection`` users.
        self._in_transaction = False
        self._tx_owner = None
        if self._protocol is None:
            return
        protocol = self._protocol
        self._protocol = None
        self._db_id = None
        protocol.close()
        # Narrow the suppression: a bounded wait on the transport
        # drain can legitimately raise TimeoutError (slow peer) or
        # OSError (already-closed writer). Anything else — especially
        # CancelledError from an outer ``asyncio.timeout`` scope — must
        # propagate so structured-concurrency cancellation semantics
        # remain intact. DEBUG-log unexpected Exceptions for
        # diagnostics; do not swallow.
        try:
            await asyncio.wait_for(protocol.wait_closed(), timeout=self._close_timeout)
        except OSError:
            # OSError subsumes TimeoutError, so the single OSError
            # entry covers the slow-peer / already-closed-writer
            # cases.
            pass
        except Exception:
            logger.debug(
                "close: unexpected drain error for %s",
                self._address,
                exc_info=True,
            )

    async def _abort_protocol(self) -> None:
        """Tear down a half-open protocol during a connect failure path.

        Close the writer, then give ``wait_closed`` the same bounded
        drain budget ``close()`` uses. Both sites share the same
        reasoning: the socket is already closed on our side, and the
        best-effort drain must not stall when the peer is
        unresponsive.
        """
        protocol = self._protocol
        if protocol is None:
            return
        self._protocol = None
        protocol.close()
        # Narrow the suppression: a bounded wait on the transport drain
        # can legitimately raise TimeoutError (slow peer) or OSError
        # (already-closed writer). Anything else — especially
        # CancelledError from an outer ``asyncio.timeout`` scope — must
        # propagate so structured-concurrency cancellation semantics
        # remain intact. DEBUG-log an unexpected Exception for
        # diagnostics; do not swallow.
        try:
            await asyncio.wait_for(protocol.wait_closed(), timeout=self._close_timeout)
        except OSError:
            # OSError subsumes TimeoutError, so the single OSError
            # entry covers the slow-peer / already-closed-writer
            # cases.
            pass
        except Exception:
            logger.debug(
                "_abort_protocol: unexpected drain error for %s",
                self._address,
                exc_info=True,
            )

    async def __aenter__(self) -> "DqliteConnection":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
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
            current_repr = repr(asyncio.current_task())
            raise InterfaceError(
                "Cannot perform operation: another operation is in progress on this "
                f"connection (current task: {current_repr}). DqliteConnection does "
                "not support concurrent coroutine access. Use a ConnectionPool to "
                "manage multiple concurrent operations."
            )
        if self._in_transaction and self._tx_owner is not None:
            current = asyncio.current_task()
            if current is not self._tx_owner:
                owner_repr = repr(self._tx_owner)
                current_repr = repr(current)
                raise InterfaceError(
                    "Cannot perform operation: connection is in a transaction owned "
                    f"by another task (owner: {owner_repr}, current: {current_repr}). "
                    "Each task should use its own connection from "
                    "the pool. Note: wrapping a connection call in "
                    "``asyncio.shield(conn.execute(...))`` creates a new task and "
                    "trips this check — shield the entire "
                    "``async with conn.transaction():`` block instead."
                )

    def _invalidate(self, cause: BaseException | None = None) -> None:
        """Mark the connection as broken after an unrecoverable error.

        If ``cause`` is provided, it is remembered so a later caller that
        hits "Not connected" can chain it as ``__cause__`` for diagnostics.

        Also clears the ``_in_use`` slot: invalidation can be invoked
        out-of-band (e.g. scheduled from the dbapi sync-timeout path
        via ``call_soon_threadsafe``), bypassing ``_run_protocol``'s
        finally that normally resets the flag. An invalidated connection
        is no longer holding a meaningful in-flight operation, so the
        flag and the liveness state must stay consistent — otherwise
        the next call deterministically raises "another operation is
        in progress" on a connection that is in fact dead.

        Synchronous writer-close + async bounded drain: ``protocol.close()``
        is synchronous (writer.close()), but ``wait_closed()`` is a
        coroutine. Without a drain, a subsequent ``close()`` early-
        returns on ``_protocol is None`` and the reader task that
        ``asyncio.open_connection`` spawned stays pending until GC,
        producing the familiar ``"Task was destroyed but it is pending"``
        noise on interpreter exit. Schedule a bounded drain task and
        remember it on ``self`` so ``close()`` can await it.
        """
        if self._protocol is not None:
            proto = self._protocol
            # Connection may already be broken; suppress close errors
            with contextlib.suppress(Exception):
                proto.close()
            # Schedule a bounded drain so close() can observe and await
            # the reader task's teardown even though _protocol is about
            # to be nulled below. Only scheduled when a running loop is
            # available — some callers (tests, inline error paths) run
            # _invalidate outside a loop; the drain is best-effort.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:

                async def _bounded_drain() -> None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(proto.wait_closed(), timeout=self._close_timeout)

                # Strong-ref on self so the task is not GC'd before
                # close() awaits it.
                self._pending_drain = loop.create_task(_bounded_drain())
        self._protocol = None
        self._db_id = None
        self._in_use = False
        # Atomic clear of transaction-state flags: an external
        # invalidation (heartbeat path, KeyboardInterrupt mid-yield,
        # ``call_soon_threadsafe``) lands here without going through
        # ``transaction()``'s ``finally`` clause, leaving stale
        # ``_in_transaction=True`` and ``_tx_owner=<dead task>`` flags
        # that ``_check_in_use`` reads. The next user (typically from
        # the pool) sees a misleading "owned by another task" rejection.
        # ``transaction()``'s finally is idempotent on the happy path;
        # this clear is load-bearing on the external-invalidation path.
        self._in_transaction = False
        self._tx_owner = None
        # Preserve the FIRST cause: ``_ensure_connected`` raises a
        # synthetic ``DqliteConnectionError("Not connected")`` chained
        # from ``self._invalidation_cause`` whenever an
        # already-invalidated connection is touched. ``_run_protocol``
        # then catches that synthetic wrapper and calls
        # ``_invalidate(synthetic_wrapper)``, which would overwrite
        # ``_invalidation_cause`` with the wrapper itself. After several
        # invalidated calls the stored cause becomes a self-chaining
        # "Not connected → Not connected → ..." stack with the real
        # transport error buried or dropped from the surface chain. The
        # ``cause is None`` guard preserves the original root cause for
        # operators triaging a leader flip.
        if cause is not None and self._invalidation_cause is None:
            self._invalidation_cause = cause

    @staticmethod
    def _validate_params(params: Sequence[Any] | None) -> None:
        """Reject non-sequence / scalar-iterable param containers.

        The qmark paramstyle wants an ordered sequence of positional
        binds. Five shapes are actively dangerous if allowed through
        (``execute("?", <shape>)``):

        * ``str`` / ``bytes`` — iterate as single chars/bytes, silently
          binding N scalars where the caller meant one value.
        * ``bytearray`` / ``memoryview`` — same shape as ``bytes`` but
          writable; same single-byte-per-bind footgun.
        * ``Mapping`` — insertion-ordered in CPython 3.7+, but the
          qmark paramstyle is positional, not named.
        * ``set`` / ``frozenset`` — iterate in unordered fashion, so
          bindings vary across Python runs.

        Previously this validator only rejected ``str | bytes``,
        letting the three remaining shapes silently scramble bindings
        at the bind layer. Match the richer dbapi-layer check
        (``_reject_non_sequence_params``) so callers going direct to
        the client layer get the same safety net.
        """
        if params is None:
            return
        # Use ``DataError`` (a DqliteError subclass) so the client
        # contract "every error is a DqliteError" holds. Callers
        # catching ``except TypeError`` previously saw a bare
        # TypeError leak past the DqliteError hierarchy.
        if isinstance(params, (str, bytes, bytearray, memoryview)):
            raise DataError(
                f"params must be a list or tuple, not {type(params).__name__!r}; "
                f"did you mean [value]?"
            )
        if isinstance(params, Mapping):
            raise DataError(
                "qmark paramstyle requires a sequence; got a mapping. "
                "Use a list or tuple positionally matching the ? placeholders."
            )
        if isinstance(params, (set, frozenset)):
            raise DataError(
                "qmark paramstyle requires an ordered sequence; got a set — "
                "iteration order is non-deterministic across runs."
            )

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
            elif _primary_sqlite_code(e.code) in _TX_AUTO_ROLLBACK_PRIMARY_CODES:
                # Server-side SQLite engine auto-rolled-back the
                # transaction; the connection itself remains healthy.
                # Clear the local tx flags so ``in_transaction`` does
                # not lie to downstream layers (PEP 249 dbapi, SA
                # dialect) and so the next statement does not run
                # under a stale "we're still inside the user's tx"
                # assumption.
                self._in_transaction = False
                self._tx_owner = None
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

        Sniffs the SQL prefix after a successful exec to keep the
        ``_in_transaction`` flag in sync with raw ``BEGIN`` / ``COMMIT``
        / ``ROLLBACK`` statements. The ``transaction()`` context
        manager updates these flags eagerly; the sniff makes the same
        invariant hold for the stdlib ``sqlite3``-style idiom of
        ``cursor.execute("BEGIN")`` ... ``cursor.execute("COMMIT")``.
        Without the sniff, ``in_transaction`` would lie about the
        engine state for raw-tx users.
        """
        self._validate_params(params)
        result = await self._run_protocol(lambda p, db: p.exec_sql(db, sql, params))
        self._update_tx_flags_from_sql(sql)
        return result

    def _update_tx_flags_from_sql(self, sql: str) -> None:
        """Update _in_transaction / _tx_owner after a successful
        execute, based on the leading verb of the SQL.

        The check is a cheap prefix sniff — comparable to stdlib
        ``sqlite3.Connection``'s autocommit logic — not a full SQL
        parser. It correctly tracks the common shapes (BEGIN /
        BEGIN TRANSACTION / BEGIN DEFERRED / BEGIN IMMEDIATE /
        BEGIN EXCLUSIVE / COMMIT / END / ROLLBACK) but cannot detect
        every embedded transaction-control statement (e.g. multi-
        statement strings). SAVEPOINT / RELEASE / ROLLBACK TO are
        deliberately ignored — they leave the outer transaction's
        boundary unchanged.
        """
        # Strip leading whitespace and a possible trailing semicolon
        # so the simple prefix match handles the common shapes.
        head = sql.lstrip()
        if not head:
            return
        upper = head.upper()
        if upper.startswith("BEGIN") and (
            len(upper) == len("BEGIN") or not upper[len("BEGIN")].isalnum()
        ):
            if not self._in_transaction:
                self._in_transaction = True
                # Deliberately leave ``_tx_owner`` as None for a raw
                # BEGIN: the dbapi's sync ``_run_sync`` submits each
                # call as a fresh task on the background loop, so
                # binding ``_tx_owner`` to the BEGIN-task would cause
                # the next sync ``execute`` call to be rejected as
                # "owned by another task". The async ``transaction()``
                # context manager (which DOES set ``_tx_owner``) keeps
                # the cross-task guard for its scope; raw BEGIN trusts
                # the caller to serialise their own access (matches
                # stdlib ``sqlite3`` semantics).
            return
        if upper.startswith("ROLLBACK"):
            after = upper[len("ROLLBACK") :].lstrip(" ;")
            # ``ROLLBACK TO`` / ``ROLLBACK TO SAVEPOINT`` only unwinds
            # to the named savepoint; the outer transaction stays open.
            if after.startswith("TO"):
                return
            if self._in_transaction:
                self._in_transaction = False
                self._tx_owner = None
            return
        if upper.startswith("COMMIT") or upper.startswith("END"):
            # ``COMMIT`` / ``END`` close the outer transaction. Both
            # forms (``COMMIT``, ``COMMIT TRANSACTION``, ``END``,
            # ``END TRANSACTION``) end here.
            verb = "COMMIT" if upper.startswith("COMMIT") else "END"
            if len(upper) == len(verb) or not upper[len(verb)].isalnum():
                if self._in_transaction:
                    self._in_transaction = False
                    self._tx_owner = None
                return

    async def query_raw(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query and return raw (column_names, rows).

        Unlike fetch() which returns dicts, this returns the raw tuple
        of (column_names, rows) from the wire protocol. Intended for
        DBAPI cursor implementations that need column names separately.

        See ``query_raw_typed`` when per-column wire ``ValueType`` codes
        are also needed (used by ``cursor.description``).
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))

    async def query_raw_typed(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """Execute a query and return (column_names, column_types, row_types, rows).

        ``column_types`` are per-column wire ``ValueType`` integer tags
        from the first response frame — suitable for populating DBAPI
        ``cursor.description[i][1]`` (``type_code``). ``row_types`` is
        one list of wire tags per decoded row; SQLite is dynamically
        typed, so different rows in the same column can carry
        different wire types (under UNION, ``CASE``, ``COALESCE``,
        ``typeof()``), and callers applying result-side converters
        need the per-row list rather than a collapsed first-row view.
        See ``dqlitewire.ValueType`` for the full enum. Use
        ``query_raw`` when type codes are not needed.
        """
        self._validate_params(params)
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
        """Context manager for transactions.

        Issues a bare ``BEGIN`` (SQLite default ``BEGIN DEFERRED``) on
        entry, ``COMMIT`` on clean exit, and ``ROLLBACK`` if the body
        raises. dqlite's Raft FSM serializes transactions across the
        cluster regardless of the ``DEFERRED`` / ``IMMEDIATE`` /
        ``EXCLUSIVE`` qualifier — isolation is always SERIALIZABLE —
        so the qualifier choice has no semantic effect on dqlite. The
        ``BEGIN`` literal matches the C/Go peer behaviour.

        Cancellation contract:
        - Cancellation during BEGIN: state cleared, CancelledError
          propagates.
        - Cancellation during the body: ROLLBACK is attempted. If
          ROLLBACK itself is cancelled, the connection is invalidated
          and CancelledError propagates (structured-concurrency
          contract — TaskGroup / asyncio.timeout() require this).
        - Cancellation during COMMIT: connection invalidated
          (server-side state ambiguous), CancelledError propagates.
        - Cancellation during ROLLBACK (body already raised): connection
          invalidated, CancelledError propagates and supersedes the
          body exception (Python chains it via ``__context__``).

        Non-cancellation ROLLBACK failure: connection is
        invalidated so the pool discards it instead of reusing a
        Python-side "_in_transaction=False" connection with live
        server-side transaction state.

        Shielding caveat: ``asyncio.shield(conn.execute(...))`` inside
        the body creates a new task whose ``current_task()`` is not
        ``_tx_owner``. ``_check_in_use`` rejects the shielded call
        with ``InterfaceError("owned by another task")``. Shield the
        **entire** ``async with conn.transaction():`` block if you
        need defensive-rollback semantics against outer cancellation
        — not individual operations inside it.
        """
        # Only the SAME task re-entering an open transaction is "nested".
        # A sibling task hitting an in-progress transaction should see
        # the "owned by another task" diagnostic so the actual remedy —
        # acquire a separate connection from the pool — is obvious. The
        # nested-tx message historically fired for sibling-task usage
        # too and pointed users at SAVEPOINT, which was the wrong
        # guidance for that pattern.
        if self._in_transaction:
            if self._tx_owner is asyncio.current_task():
                raise InterfaceError(
                    "Nested transactions are not supported; use SAVEPOINT directly"
                )
            owner_repr = repr(self._tx_owner)
            raise InterfaceError(
                "Cannot start transaction: connection is in a transaction owned "
                f"by another task ({owner_repr}). Each task should use its own "
                "connection from the pool."
            )

        # Set the flags before the BEGIN await — the early set is a
        # secondary guard atop ``_run_protocol``'s ``_in_use`` flag.
        # While ``_in_use`` rejects concurrent calls with "another
        # operation in progress", the early ``_in_transaction=True``
        # makes a *task switch mid-BEGIN* surface as the more specific
        # "nested transactions are not supported" — useful when
        # callers wrap operations in a single shared connection by
        # mistake. The except clause clears on any failure path.
        self._in_transaction = True
        self._tx_owner = asyncio.current_task()
        try:
            await self.execute(_TRANSACTION_BEGIN_SQL)
        except BaseException:
            self._tx_owner = None
            self._in_transaction = False
            raise

        commit_attempted = False
        try:
            yield
            commit_attempted = True
            await self.execute(_TRANSACTION_COMMIT_SQL)
        except BaseException:
            if commit_attempted:
                # COMMIT was sent but failed. Server-side state is ambiguous
                # (maybe committed, maybe still open, maybe rolled back). We
                # cannot safely reuse this connection — invalidate so the
                # pool discards it instead of recycling an unknown state.
                self._invalidate()
            else:
                # Body raised before COMMIT; try to roll back.
                #
                # Narrow suppression to Exception (NOT BaseException):
                # CancelledError / KeyboardInterrupt / SystemExit must
                # propagate. Previously ``suppress(BaseException)``
                # swallowed cancellation, breaking structured-concurrency
                # contracts.
                #
                # If ROLLBACK fails for any reason (including the narrow
                # cancellation catch below), the connection's transaction
                # state is unknowable from our side and the connection
                # must be invalidated so the pool discards it on return
                # from our side. The original body exception is still the
                # one that propagates, except for cancellation which
                # takes precedence.
                try:
                    await self.execute(_TRANSACTION_ROLLBACK_SQL)
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    # Rollback interrupted mid-flight. Server-side tx is
                    # in an unknown state; invalidate and propagate the
                    # higher-priority signal.
                    logger.debug(
                        "transaction(address=%s, id=%s): rollback was "
                        "cancelled mid-flight; connection invalidated; "
                        "propagating cancellation",
                        self._address,
                        id(self),
                        exc_info=True,
                    )
                    self._invalidate()
                    raise
                except OperationalError as roll_exc:
                    # SQLITE_ERROR with the "no transaction is active"
                    # wording is the deterministic "nothing to roll
                    # back" reply — the body exception aborted before
                    # the implicit BEGIN reached the server, or the
                    # server already auto-rolled-back. The connection
                    # is healthy; preserve it and re-raise the body
                    # exception. Mirrors the dbapi layer's
                    # _NO_TX_CODES whitelist (ISSUE-696).
                    if _is_no_tx_rollback_error(roll_exc):
                        logger.debug(
                            "transaction(address=%s, id=%s): rollback "
                            "found no active transaction (server-side "
                            "tx already gone); preserving connection",
                            self._address,
                            id(self),
                        )
                    else:
                        logger.debug(
                            "transaction(address=%s, id=%s): rollback failed "
                            "with OperationalError; connection invalidated; "
                            "propagating original body exception",
                            self._address,
                            id(self),
                            exc_info=True,
                        )
                        self._invalidate()
                except Exception:
                    # Rollback failed for a non-OperationalError reason.
                    # Invalidate so the pool discards on return, then
                    # re-raise the ORIGINAL body exception (below)
                    # — rollback failure is a secondary concern.
                    logger.debug(
                        "transaction(address=%s, id=%s): rollback failed; "
                        "connection invalidated; propagating original body "
                        "exception",
                        self._address,
                        id(self),
                        exc_info=True,
                    )
                    self._invalidate()
            raise
        finally:
            self._tx_owner = None
            self._in_transaction = False
