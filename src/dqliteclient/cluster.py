"""Leader discovery, connect-with-retry, and cluster administration."""

import asyncio
import contextlib
import ipaddress
import logging
import math
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, NoReturn, final

from dqliteclient._dial import DialFunc, open_connection
from dqliteclient._validate import (
    DEFAULT_CLOSE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    get_current_pid,
    parse_address,
    validate_max_attempts,
    validate_timeout,
)
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, NodeStore
from dqliteclient.protocol import DqliteProtocol, validate_positive_int_or_none
from dqliteclient.retry import retry_with_backoff
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    NodeRole,
    sanitize_for_log,
)
from dqlitewire import NodeInfo as WireNodeInfo

__all__ = [
    "ClusterClient",
    "LeaderInfo",
    "NodeMetadata",
    "RedirectPolicy",
    "allowlist_policy",
    "default_safe_redirect_policy",
]

logger = logging.getLogger(__name__)

type RedirectPolicy = Callable[[str], bool]
"""Return True to accept an address a peer hands back (leader hints, membership lists)."""

_DEFAULT_CONCURRENT_PROBES: Final[int] = 10


def _timeout_message(safe_address: str, phase: str, owner: "ClusterClient") -> str:
    if phase == "dial":
        return f"Connection to {safe_address} timed out (dial_timeout={owner._dial_timeout}s)"
    return f"Handshake with {safe_address} timed out (attempt_timeout={owner._attempt_timeout}s)"


_DEFAULT_CONNECT_ATTEMPTS: Final[int] = 3
_PROBE_DRAIN_SECONDS: Final[float] = 0.1
_RETRYABLE: Final = (DqliteConnectionError, ClusterError, OSError)
_PROBE_FAILURES: Final = (
    DqliteConnectionError,
    ProtocolError,
    OperationalError,
    OSError,
    TimeoutError,
    ValueError,
)
_random = random.SystemRandom()


@final
@dataclass(frozen=True, slots=True)
class LeaderInfo:
    node_id: int
    address: str


@final
@dataclass(frozen=True, slots=True)
class NodeMetadata:
    failure_domain: int
    weight: int


def _same_address(a: str, b: str) -> bool:
    try:
        return parse_address(a) == parse_address(b)
    except ValueError:
        return a == b


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class ClusterClient:
    """Finds the leader among the nodes of a :class:`NodeStore` and talks to it."""

    def __init__(
        self,
        node_store: NodeStore,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        concurrent_leader_conns: int = _DEFAULT_CONCURRENT_PROBES,
        redirect_policy: RedirectPolicy | None = None,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        max_message_size: int | None = None,
        trust_server_heartbeat: bool = False,
        dial_func: DialFunc | None = None,
    ) -> None:
        validate_timeout(timeout)
        if dial_timeout is not None:
            validate_timeout(dial_timeout, name="dial_timeout")
        if attempt_timeout is not None:
            validate_timeout(attempt_timeout, name="attempt_timeout")
        if not _is_int(concurrent_leader_conns):
            raise TypeError(
                f"concurrent_leader_conns must be int, got {type(concurrent_leader_conns).__name__}"
            )
        if concurrent_leader_conns < 1:
            raise ValueError(f"concurrent_leader_conns must be >= 1, got {concurrent_leader_conns}")
        if max_message_size is not None:
            if not _is_int(max_message_size):
                raise TypeError(
                    f"max_message_size must be int or None, got {type(max_message_size).__name__}"
                )
            if max_message_size < 1:
                raise ValueError(f"max_message_size must be >= 1, got {max_message_size}")
        self._node_store = node_store
        self._timeout = timeout
        self._dial_timeout = dial_timeout if dial_timeout is not None else timeout
        self._attempt_timeout = attempt_timeout if attempt_timeout is not None else timeout
        self._concurrent = concurrent_leader_conns
        self._redirect_policy = redirect_policy
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._max_message_size = max_message_size
        self._trust_server_heartbeat = trust_server_heartbeat
        self._dial_func = dial_func
        self._last_leader: str | None = None
        self._sweeps: dict[tuple[bool, RedirectPolicy | None], asyncio.Task[str]] = {}
        self._pid = get_current_pid()

    @classmethod
    def from_addresses(
        cls,
        addresses: Sequence[str],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        concurrent_leader_conns: int = _DEFAULT_CONCURRENT_PROBES,
        redirect_policy: RedirectPolicy | None = None,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        max_message_size: int | None = None,
        trust_server_heartbeat: bool = False,
        dial_func: DialFunc | None = None,
    ) -> "ClusterClient":
        return cls(
            MemoryNodeStore(addresses),
            timeout=timeout,
            dial_timeout=dial_timeout,
            attempt_timeout=attempt_timeout,
            concurrent_leader_conns=concurrent_leader_conns,
            redirect_policy=redirect_policy,
            max_total_rows=max_total_rows,
            max_continuation_frames=max_continuation_frames,
            max_message_size=max_message_size,
            trust_server_heartbeat=trust_server_heartbeat,
            dial_func=dial_func,
        )

    def _check_pid(self) -> None:
        if get_current_pid() != self._pid:
            raise InterfaceError(
                f"ClusterClient used after fork; reconstruct it in the child process "
                f"(created in pid {self._pid}, current pid {get_current_pid()})"
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"cannot pickle {type(self).__name__!r}: it owns live connections")

    def _check_policy(self, address: str, policy: RedirectPolicy | None) -> None:
        effective = policy if policy is not None else self._redirect_policy
        if effective is not None and not effective(address):
            raise ClusterPolicyError(
                f"redirect to {sanitize_for_log(address)!r} rejected by the redirect policy"
            )

    # -- leader discovery ------------------------------------------------------

    async def find_leader(
        self, *, trust_server_heartbeat: bool = False, policy: RedirectPolicy | None = None
    ) -> str:
        """Return the leader's address. Concurrent callers share one sweep.

        Raises :class:`ClusterError` when no node reports a leader and
        :class:`ClusterPolicyError` when a peer hands back a rejected address.
        """
        self._check_pid()
        key = (trust_server_heartbeat, policy)
        sweep = self._sweeps.get(key)
        if sweep is None:
            sweep = asyncio.create_task(self._sweep(trust_server_heartbeat, policy))
            self._sweeps[key] = sweep
            sweep.add_done_callback(lambda t: self._sweeps.pop(key, None))
        try:
            return await asyncio.shield(sweep)
        except asyncio.CancelledError:
            if not sweep.done():
                sweep.add_done_callback(_observe)
            raise

    async def _sweep(self, trust: bool, policy: RedirectPolicy | None) -> str:
        cached = self._last_leader
        if cached is not None:
            self._last_leader = None
            leader = await self._probe(cached, trust, policy, [])
            if leader is not None:
                self._last_leader = leader
                return leader
        async with asyncio.timeout(self._dial_timeout):
            nodes = list(await self._node_store.get_nodes())
        if not nodes:
            raise ClusterError("No nodes configured")
        _random.shuffle(nodes)
        nodes.sort(key=lambda n: int(n.role))
        failures: list[str] = []
        semaphore = asyncio.Semaphore(self._concurrent)

        async def probe(node: NodeInfo) -> str | None:
            async with semaphore:
                return await self._probe(node.address, trust, policy, failures)

        tasks = [asyncio.create_task(probe(node)) for node in nodes]
        try:
            for done in asyncio.as_completed(tasks):
                leader = await done
                if leader is not None:
                    self._last_leader = leader
                    return leader
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        raise ClusterError(f"Could not find leader. Errors: {'; '.join(failures)}")

    async def _probe(
        self, address: str, trust: bool, policy: RedirectPolicy | None, failures: list[str]
    ) -> str | None:
        """Ask ``address`` who leads; verify a redirect before trusting it. ``None`` on miss."""
        safe = sanitize_for_log(address)
        try:
            async with asyncio.timeout(self._attempt_timeout):
                hint = await self._query_leader(address, trust)
        except _PROBE_FAILURES as exc:
            failures.append(f"{safe}: {sanitize_for_log(str(exc))}")
            logger.debug("find_leader: probe of %s failed: %s", safe, failures[-1])
            return None
        if hint is None:
            failures.append(f"{safe}: no leader known")
            logger.debug("find_leader: %s knows no leader", safe)
            return None
        self._check_policy(hint, policy)
        if _same_address(hint, address):
            return hint
        with contextlib.suppress(*_PROBE_FAILURES):
            async with asyncio.timeout(self._attempt_timeout):
                confirmed = await self._query_leader(hint, trust)
            if confirmed is not None and _same_address(confirmed, hint):
                return hint
        failures.append(f"{safe}: redirect to {sanitize_for_log(hint)} did not confirm")
        return None

    async def _query_leader(self, address: str, trust: bool) -> str | None:
        """One version-only handshake plus a LEADER request. ``None`` when no leader is known."""
        async with self._admin_protocol(address, trust, register=False) as protocol:
            node_id, leader = await protocol.get_leader()
        if not leader:
            return None
        if node_id == 0:
            raise ProtocolError(
                f"malformed LEADER reply from {sanitize_for_log(address)}: "
                "node_id=0 with an address"
            )
        return leader

    @contextlib.asynccontextmanager
    async def _admin_protocol(
        self, address: str, trust: bool, *, register: bool
    ) -> AsyncIterator[DqliteProtocol]:
        safe = sanitize_for_log(address)
        protocol: DqliteProtocol | None = None
        phase = "dial"
        try:
            async with asyncio.timeout(self._attempt_timeout):
                try:
                    async with asyncio.timeout(self._dial_timeout):
                        reader, writer = await open_connection(address, dial_func=self._dial_func)
                except TimeoutError:
                    raise
                except OSError as e:
                    raise DqliteConnectionError(f"Failed to connect to {safe}: {e}") from e
                phase = "handshake"
                protocol = DqliteProtocol(
                    reader,
                    writer,
                    timeout=self._timeout,
                    trust_server_heartbeat=trust,
                    max_total_rows=self._max_total_rows,
                    max_continuation_frames=self._max_continuation_frames,
                    max_message_size=self._max_message_size,
                    address=address,
                )
                if register:
                    await protocol.handshake()
                else:
                    await protocol.negotiate_protocol_only()
        except TimeoutError as e:
            if protocol is not None:
                protocol.close()
            raise DqliteConnectionError(_timeout_message(safe, phase, self)) from e
        except BaseException:
            if protocol is not None:
                protocol.close()
            raise
        try:
            yield protocol
        finally:
            protocol.close()
            with contextlib.suppress(OSError, TimeoutError):
                async with asyncio.timeout(_PROBE_DRAIN_SECONDS):
                    await protocol.wait_closed()

    # -- connecting --------------------------------------------------------------

    async def connect(
        self,
        database: str = "default",
        *,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        max_attempts: int | None = None,
        max_elapsed_seconds: float | None = None,
        policy: RedirectPolicy | None = None,
        max_message_size: int | None = None,
    ) -> DqliteConnection:
        """Find the leader and open a :class:`DqliteConnection` to it, with bounded
        backoff retry on transport and cluster failures."""
        validate_max_attempts(max_attempts)
        if max_elapsed_seconds is not None and (
            isinstance(max_elapsed_seconds, bool)
            or not isinstance(max_elapsed_seconds, int | float)
            or not math.isfinite(max_elapsed_seconds)
            or max_elapsed_seconds <= 0
        ):
            raise ValueError(
                f"max_elapsed_seconds must be a positive number, got {max_elapsed_seconds!r}"
            )
        message_size = max_message_size if max_message_size is not None else self._max_message_size
        total = max_attempts or _DEFAULT_CONNECT_ATTEMPTS
        attempts = 0

        async def attempt() -> DqliteConnection:
            nonlocal attempts
            attempts += 1
            try:
                return await once()
            except Exception as exc:
                logger.debug(
                    "ClusterClient.connect attempt %d/%d failed: %s: %s",
                    attempts,
                    total,
                    type(exc).__name__,
                    sanitize_for_log(str(exc)),
                )
                raise

        async def once() -> DqliteConnection:
            leader = await self.find_leader(
                trust_server_heartbeat=trust_server_heartbeat, policy=policy
            )
            try:
                parse_address(leader)
            except ValueError as exc:
                raise ClusterPolicyError(
                    f"invalid leader address {sanitize_for_log(leader)!r}: {exc}"
                ) from exc
            conn = DqliteConnection(
                leader,
                database=database,
                timeout=self._timeout,
                dial_timeout=self._dial_timeout,
                attempt_timeout=self._attempt_timeout,
                max_total_rows=max_total_rows,
                max_continuation_frames=max_continuation_frames,
                trust_server_heartbeat=trust_server_heartbeat,
                close_timeout=close_timeout,
                dial_func=self._dial_func,
                max_message_size=message_size,
            )
            try:
                await conn.connect()
            except _RETRYABLE:
                self._last_leader = None
                conn.terminate()
                raise
            return conn

        try:
            return await retry_with_backoff(
                attempt,
                max_attempts=total,
                max_delay=1.0,
                max_elapsed_seconds=max_elapsed_seconds,
                retryable_exceptions=_RETRYABLE,
                excluded_exceptions=(ClusterPolicyError,),
            )
        except ClusterPolicyError:
            raise
        except _RETRYABLE as exc:
            logger.warning(
                "cluster: connect gave up: %s: %s", type(exc).__name__, sanitize_for_log(str(exc))
            )
            raise

    # -- administration ----------------------------------------------------------

    @contextlib.asynccontextmanager
    async def open_admin_connection(self, address: str) -> AsyncIterator[DqliteProtocol]:
        """A fresh, handshaken protocol to ``address``, closed on exit."""
        self._check_pid()
        async with self._admin_protocol(address, self._trust_server_heartbeat, register=False) as p:
            yield p

    async def _leader_protocol(self, policy: RedirectPolicy | None = None) -> str:
        return await self.find_leader(policy=policy)

    async def cluster_info(self, *, policy: RedirectPolicy | None = None) -> list[WireNodeInfo]:
        """The leader's view of the membership, minus addresses the policy rejects."""
        leader = await self._leader_protocol(policy)
        try:
            async with self.open_admin_connection(leader) as protocol:
                nodes = await protocol.cluster()
        except (OperationalError, DqliteConnectionError, ProtocolError):
            self._last_leader = None
            raise
        effective = policy if policy is not None else self._redirect_policy
        if effective is None:
            return nodes
        kept = []
        for node in nodes:
            if effective(node.address):
                kept.append(node)
            else:
                logger.warning(
                    "cluster_info: dropping node %d at %s: rejected by policy",
                    node.node_id,
                    sanitize_for_log(node.address),
                )
        return kept

    async def leader_info(self, *, policy: RedirectPolicy | None = None) -> LeaderInfo | None:
        """The leader's ``(node_id, address)``, or ``None`` during an election."""
        leader = await self._leader_protocol(policy)
        try:
            async with self.open_admin_connection(leader) as protocol:
                node_id, address = await protocol.get_leader()
        except (OperationalError, DqliteConnectionError, ProtocolError):
            self._last_leader = None
            raise
        if not address:
            return None
        if node_id == 0:
            raise ProtocolError(
                f"malformed LEADER reply from {sanitize_for_log(leader)}: node_id=0 with an address"
            )
        self._check_policy(address, policy)
        return LeaderInfo(node_id=node_id, address=address)

    async def transfer_leadership(self, target_node_id: int) -> None:
        _validate_node_id(target_node_id, "target_node_id")
        await self._on_leader(lambda p: p.transfer(target_node_id), invalidate=True)

    async def add_node(
        self, node_id: int, address: str, *, role: NodeRole = NodeRole.SPARE
    ) -> None:
        """Add a node; a non-spare ``role`` is applied with a second ASSIGN request."""
        _validate_node_id(node_id)
        if not isinstance(role, NodeRole):
            raise TypeError(f"role must be a NodeRole, got {type(role).__name__}")
        _validate_address(address, "add_node")

        async def do(protocol: DqliteProtocol) -> None:
            await protocol.add(node_id, address)
            if role != NodeRole.SPARE:
                await protocol.assign(node_id, role)

        await self._on_leader(do, invalidate=True)

    async def assign_role(self, node_id: int, role: NodeRole) -> None:
        _validate_node_id(node_id)
        if not isinstance(role, NodeRole):
            raise TypeError(f"role must be a NodeRole, got {type(role).__name__}")
        await self._on_leader(lambda p: p.assign(node_id, role), invalidate=True)

    async def remove_node(self, node_id: int) -> None:
        """Remove a node. Transfer leadership away from it first if it is the leader."""
        _validate_node_id(node_id)
        await self._on_leader(lambda p: p.remove(node_id), invalidate=True)

    async def describe(self, *, address: str | None = None) -> NodeMetadata:
        """Failure domain and weight of ``address``, or of the leader."""
        target = await self._target(address, "describe")
        async with self.open_admin_connection(target) as protocol:
            response = await protocol.describe()
        return NodeMetadata(failure_domain=response.failure_domain, weight=response.weight)

    async def set_weight(self, weight: int, *, address: str | None = None) -> None:
        if not _is_int(weight):
            raise TypeError(f"weight must be int, got {type(weight).__name__}")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        target = await self._target(address, "set_weight")
        async with self.open_admin_connection(target) as protocol:
            await protocol.weight(weight)

    async def dump(self, database: str) -> dict[str, bytes]:
        """``{filename: bytes}`` of ``database`` as served by the leader."""
        if not isinstance(database, str) or not database:
            raise TypeError("database must be a non-empty str")
        leader = await self.find_leader()
        async with self.open_admin_connection(leader) as protocol:
            return await protocol.dump(database)

    async def _target(self, address: str | None, context: str) -> str:
        self._check_pid()
        if address is None:
            return await self.find_leader()
        _validate_address(address, context)
        return address

    async def _on_leader[T](
        self, fn: Callable[[DqliteProtocol], Awaitable[T]], *, invalidate: bool
    ) -> T:
        leader = await self.find_leader()
        try:
            async with self.open_admin_connection(leader) as protocol:
                return await fn(protocol)
        finally:
            if invalidate:
                self._last_leader = None


def _observe(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _validate_node_id(node_id: object, name: str = "node_id") -> None:
    if not _is_int(node_id):
        raise TypeError(f"{name} must be int, got {type(node_id).__name__}")
    if node_id < 1:  # type: ignore[operator]
        raise ValueError(f"{name} must be >= 1, got {node_id}")


def _validate_address(address: object, context: str) -> None:
    if not isinstance(address, str) or not address:
        raise TypeError(f"{context}: address must be a non-empty str")
    try:
        parse_address(address)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid address {address!r}: {exc}") from exc


def allowlist_policy(addresses: Iterable[str]) -> RedirectPolicy:
    """Accept only the given addresses, compared as canonical ``(host, port)``."""
    allowed = frozenset(parse_address(a) for a in addresses)

    def policy(address: str) -> bool:
        try:
            return parse_address(address) in allowed
        except ValueError:
            return False

    return policy


def default_safe_redirect_policy(
    *, include_rfc1918: bool = True, include_loopback: bool = False
) -> RedirectPolicy:
    """Reject link-local addresses (and loopback unless allowed); hostnames pass through.

    Applies to the literal address, so a hostname resolving to a rejected range is not
    caught; pair with :func:`allowlist_policy` for strict control.
    """

    def policy(address: str) -> bool:
        try:
            host, _ = parse_address(address)
            ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(host)
        except ValueError:
            return "@" not in address and _looks_like_address(address)
        if isinstance(ip, ipaddress.IPv6Address):
            ip = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else ip)
        if ip.is_link_local:
            return False
        if ip.is_loopback and not include_loopback:
            return False
        return include_rfc1918 or not ip.is_private

    return policy


def _looks_like_address(address: str) -> bool:
    try:
        parse_address(address)
    except ValueError:
        return False
    return True
