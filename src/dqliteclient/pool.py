"""A pool of connections to the cluster leader."""

import asyncio
import contextlib
import logging
import os
import warnings
import weakref
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, NoReturn, Self

from dqliteclient._dial import DialFunc
from dqliteclient._validate import (
    CLOSE_TIMEOUT_FLOOR,
    CLOSE_TIMEOUT_FLOOR_RATIONALE,
    DEFAULT_CLOSE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    get_current_pid,
    validate_max_attempts,
    validate_timeout,
)
from dqliteclient.cluster import ClusterClient, RedirectPolicy
from dqliteclient.connection import DqliteConnection, is_no_transaction_error
from dqliteclient.exceptions import DqliteConnectionError, DqliteError, InterfaceError
from dqliteclient.node_store import MemoryNodeStore, NodeStore
from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES, DEFAULT_MAX_TOTAL_ROWS

__all__ = ["ConnectionPool"]

logger = logging.getLogger(__name__)


def _warn_if_unclosed(flags: list[bool], pid: int) -> None:
    if flags[0] or os.getpid() != pid:
        return
    warnings.warn(
        "ConnectionPool was garbage-collected without await close(); "
        "call close() to release its connections promptly.",
        ResourceWarning,
        stacklevel=2,
    )


class ConnectionPool:
    """Idle-queue pool of leader connections. Not thread-safe; one event loop only."""

    def __init__(
        self,
        addresses: Sequence[str] | None = None,
        *,
        database: str = "default",
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        cluster: ClusterClient | None = None,
        node_store: NodeStore | None = None,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        max_message_size: int | None = None,
        max_attempts: int | None = None,
        max_elapsed_seconds: float | None = None,
        dial_func: DialFunc | None = None,
        concurrent_leader_conns: int | None = None,
        redirect_policy: RedirectPolicy | None = None,
    ) -> None:
        for name, value in (("min_size", min_size), ("max_size", max_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if not 0 <= min_size <= max_size:
            raise ValueError(f"min_size must be between 0 and max_size, got {min_size}")
        validate_timeout(timeout)
        validate_timeout(
            close_timeout,
            name="close_timeout",
            min_value=CLOSE_TIMEOUT_FLOOR,
            min_value_rationale=CLOSE_TIMEOUT_FLOOR_RATIONALE,
        )
        validate_max_attempts(max_attempts)
        if sum(x is not None for x in (cluster, node_store, addresses)) > 1:
            raise ValueError("pass at most one of addresses, node_store, or cluster")
        if cluster is not None:
            owned: dict[str, object] = {
                "dial_func": dial_func,
                "concurrent_leader_conns": concurrent_leader_conns,
                "redirect_policy": redirect_policy,
            }
            for owned_name, owned_value in owned.items():
                if owned_value is not None:
                    raise ValueError(
                        f"{owned_name} cannot be combined with cluster= "
                        "(the cluster owns its settings)"
                    )
        if cluster is None:
            store = node_store if node_store is not None else MemoryNodeStore(addresses)
            cluster = ClusterClient(
                store,
                timeout=timeout,
                dial_timeout=dial_timeout,
                attempt_timeout=attempt_timeout,
                concurrent_leader_conns=concurrent_leader_conns or 10,
                redirect_policy=redirect_policy,
                max_total_rows=max_total_rows,
                max_continuation_frames=max_continuation_frames,
                max_message_size=max_message_size,
                trust_server_heartbeat=trust_server_heartbeat,
                dial_func=dial_func,
            )
        self._cluster = cluster
        self._connect_options: dict[str, Any] = {
            "max_total_rows": max_total_rows,
            "max_continuation_frames": max_continuation_frames,
            "trust_server_heartbeat": trust_server_heartbeat,
            "close_timeout": close_timeout,
            "max_attempts": max_attempts,
            "max_elapsed_seconds": max_elapsed_seconds,
            "max_message_size": max_message_size,
        }
        self._database = database
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._idle: list[DqliteConnection] = []
        self._size = 0
        self._condition = asyncio.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._initialized = False
        self._closed_flag = [False]
        self._pid = get_current_pid()
        self._finalizer = weakref.finalize(self, _warn_if_unclosed, self._closed_flag, self._pid)

    @property
    def closed(self) -> bool:
        return self._closed_flag[0]

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"idle={len(self._idle)} size={self._size}"
        return f"<ConnectionPool max_size={self._max_size} {state} at 0x{id(self):x}>"

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"cannot pickle {type(self).__name__!r}: it owns live sockets")

    def _check_usable(self) -> None:
        if get_current_pid() != self._pid:
            raise InterfaceError(
                f"Pool used after fork; reconstruct it in the child process "
                f"(created in pid {self._pid}, current pid {get_current_pid()})"
            )
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise InterfaceError("ConnectionPool is bound to a different event loop")
        if self.closed:
            raise DqliteConnectionError(f"Pool is closed (id={id(self)})")

    async def _create_connection(self) -> DqliteConnection:
        return await self._cluster.connect(self._database, **self._connect_options)

    async def initialize(self) -> None:
        """Open ``min_size`` connections up front. Idempotent."""
        self._check_usable()
        if self._initialized:
            return
        self._initialized = True
        async with self._condition:
            wanted = max(0, self._min_size - self._size)
            self._size += wanted
        results = await asyncio.gather(
            *(self._create_connection() for _ in range(wanted)), return_exceptions=True
        )
        opened = [r for r in results if isinstance(r, DqliteConnection)]
        failures = [r for r in results if isinstance(r, BaseException)]
        async with self._condition:
            if failures or self.closed:
                self._size -= wanted
            else:
                self._idle.extend(opened)
            self._condition.notify_all()
        if failures or self.closed:
            self._initialized = False
            await asyncio.gather(*(c.close() for c in opened), return_exceptions=True)
            for failure in failures:
                if isinstance(failure, asyncio.CancelledError | KeyboardInterrupt | SystemExit):
                    raise failure
            if failures:
                for failure in failures:
                    logger.warning(
                        "pool.initialize: _create_connection failed: %s: %s",
                        type(failure).__name__,
                        failure,
                    )
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup(
                    f"pool.initialize: {len(failures)} of {wanted} connects failed", failures
                )
            raise DqliteConnectionError(f"Pool is closed (id={id(self)})")

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Check a connection out for the duration of the block."""
        conn = await self._acquire()
        try:
            yield conn
        finally:
            await self._release(conn)

    async def _acquire(self) -> DqliteConnection:
        self._check_usable()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        async with self._condition:
            while True:
                if self.closed:
                    raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
                while self._idle:
                    conn = self._idle.pop()
                    if conn.is_connected:
                        return conn
                    self._size -= 1
                    conn.terminate()
                if self._size < self._max_size:
                    self._size += 1
                    break
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise DqliteConnectionError(
                        "Timed out waiting for a connection from the pool "
                        f"(max_size={self._max_size}, "
                        f"checked_out={self._size - len(self._idle)}, timeout={self._timeout}s)"
                    )
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(remaining):
                        await self._condition.wait()
        try:
            async with asyncio.timeout(max(deadline - loop.time(), 0.001)):
                return await self._create_connection()
        except BaseException as exc:
            async with self._condition:
                self._size -= 1
                self._condition.notify()
            if isinstance(exc, TimeoutError):
                raise DqliteConnectionError(
                    "Timed out creating a fresh connection from the pool "
                    f"(timeout={self._timeout}s)"
                ) from exc
            raise

    async def _release(self, conn: DqliteConnection) -> None:
        if get_current_pid() != self._pid:
            return
        keep = await self._reset(conn)
        async with self._condition:
            if keep and not self.closed:
                self._idle.append(conn)
            else:
                keep = False
                self._size -= 1
            self._condition.notify()
        if not keep:
            await self._close_quietly(conn)

    async def _reset(self, conn: DqliteConnection) -> bool:
        """Roll back a possibly open transaction; False if the connection should be dropped."""
        if not conn.is_connected:
            return False
        if conn.in_transaction:
            try:
                await conn.execute("ROLLBACK")
            except DqliteError as exc:
                if not is_no_transaction_error(exc):
                    logger.debug("pool: dropping connection after ROLLBACK failed: %s", exc)
                    return False
            except Exception:
                return False
        return conn.is_connected

    async def _close_quietly(self, conn: DqliteConnection) -> None:
        try:
            await conn.close()
        except Exception:
            logger.debug("pool: close failed", exc_info=True)
            conn.terminate()

    async def close(self) -> None:
        """Close idle connections; checked-out ones close when returned. Idempotent."""
        if self.closed:
            return
        self._closed_flag[0] = True
        self._finalizer.detach()
        if get_current_pid() != self._pid:
            return
        async with self._condition:
            idle, self._idle = self._idle, []
            self._size -= len(idle)
            self._condition.notify_all()
        await asyncio.gather(*(self._close_quietly(c) for c in idle), return_exceptions=True)

    # -- convenience -------------------------------------------------------------

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        async with self.acquire() as conn:
            return await conn.execute(sql, params)

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        async with self.acquire() as conn:
            return await conn.fetch(sql, params)

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        async with self.acquire() as conn:
            return await conn.fetchone(sql, params)

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        async with self.acquire() as conn:
            return await conn.fetchall(sql, params)

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        async with self.acquire() as conn:
            return await conn.fetchval(sql, params)

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
