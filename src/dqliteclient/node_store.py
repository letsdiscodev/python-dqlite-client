"""Node store interfaces for cluster discovery."""

import asyncio
import contextlib
import logging
import os
import stat
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn, Protocol, final, runtime_checkable

from dqliteclient import connection as _conn_mod
from dqliteclient.exceptions import ClusterError, InterfaceError
from dqliteclient.protocol import _is_int_not_bool
from dqlitewire import NodeRole
from dqlitewire import sanitize_server_text as _sanitize_display_text

__all__ = ["MemoryNodeStore", "NodeInfo", "NodeStore", "YamlNodeStore"]

logger = logging.getLogger(__name__)

# Codepoint cap for payload-derived values in YAML-loader ``ClusterError``
# diagnostics: a single oversized/corrupt entry must not flood operator logs.
_MAX_DISPLAY_VALUE = 256


def _cap_and_sanitize(value: object, n: int = _MAX_DISPLAY_VALUE) -> str:
    """Sanitise (CWE-117) and length-cap a payload-derived value for a ClusterError."""
    s = value if isinstance(value, str) else str(value)
    s = _sanitize_display_text(s)
    if len(s) <= n:
        return s
    return f"{s[:n]}... [truncated, {len(s) - n} codepoints]"


# Cap on cancel-storm absorption in set_nodes's shield-drain loop: the lock is
# held across it, so a wedged worker thread would otherwise wedge the store
# process-wide. Exceeding it raises RuntimeError instead of spinning forever.
_MAX_CANCEL_DRAIN_ITERS = 64


@final
@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a cluster node. Frozen so it is hashable and immutable."""

    node_id: int
    """Dqlite raft node id (1-based). In the MemoryNodeStore seeding path it is a
    synthesised placeholder (i+1), NOT the authoritative cluster-side id."""

    address: str
    """``host:port`` string; IPv6 must be bracketed (``[::1]:9000``)."""

    role: NodeRole

    def __post_init__(self) -> None:
        # Validate at construction so a degenerate NodeInfo can't reach the
        # wire-encoder. Coerce bare ints into NodeRole so an invalid value raises
        # here rather than at peer-decode time.
        if not isinstance(self.role, NodeRole):
            try:
                coerced = NodeRole(self.role)
            except ValueError as e:
                raise ValueError(
                    f"NodeInfo: unknown role {self.role!r}; valid roles are "
                    f"0 (VOTER), 1 (STANDBY), 2 (SPARE)"
                ) from e
            object.__setattr__(self, "role", coerced)
        # Reject bool (int subclass), non-int, and < 1: id 0 is the upstream
        # "no node" sentinel, so it can't be a real cluster member.
        if not _is_int_not_bool(self.node_id):
            raise TypeError(f"NodeInfo.node_id must be int, got {type(self.node_id).__name__}")
        if self.node_id < 1:
            raise ValueError(f"NodeInfo.node_id must be >= 1, got {self.node_id}")


@runtime_checkable
class NodeStore(Protocol):
    """Structural (PEP 544) interface for storing cluster node information."""

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Return an immutable snapshot of known nodes.

        MUST NOT be mutated after return: callers iterate it across await points
        without defensive copies, so a live mutable backing would cause torn reads.
        """
        ...

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update known nodes; the swap MUST be atomic w.r.t. concurrent get_nodes()."""
        ...


class MemoryNodeStore(NodeStore):
    """In-memory node store backed by an immutable tuple (atomic reference swap),
    with concurrent writers serialised by an asyncio.Lock to avoid lost updates."""

    def __init__(
        self,
        addresses: Sequence[str] | None = None,
        *,
        initial_addresses: Sequence[str] | None = None,
    ) -> None:
        """Seed with ``host:port`` addresses. ``initial_addresses`` is a deprecated alias.

        Synchronous: validating a very large seed list blocks the loop. Construct at
        process start, or keep the list small; the async set_nodes path yields.
        """
        if addresses is not None and initial_addresses is not None:
            raise TypeError("Pass only one of 'addresses' or 'initial_addresses'")
        if initial_addresses is not None and addresses is None:
            # stacklevel=2 points the warning at the caller, not this line.
            warnings.warn(
                "MemoryNodeStore(initial_addresses=...) is deprecated; "
                "use MemoryNodeStore(addresses=...) instead. "
                "The 'initial_addresses' alias will be removed in a "
                "future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        seed = addresses if addresses is not None else initial_addresses
        if seed:
            # Wrap seeds in synthetic NodeInfo (id=i+1, VOTER) and route through the
            # shared validator so strip/parse/dedup lives in one place.
            raw_nodes: list[NodeInfo] = []
            for raw in seed:
                if not isinstance(raw, str):
                    raise TypeError(
                        f"NodeStore addresses must be 'host:port' strings, got {type(raw).__name__}"
                    )
                raw_nodes.append(
                    NodeInfo(node_id=len(raw_nodes) + 1, address=raw, role=NodeRole.VOTER)
                )
            self._nodes: tuple[NodeInfo, ...] = _validate_and_normalise_nodes(raw_nodes)
        else:
            self._nodes = ()
        # Eager init: a lazy None-then-initialise shape lost mutual exclusion under
        # concurrent first-time callers (both build a fresh Lock, one update lost).
        self._set_nodes_lock: asyncio.Lock = asyncio.Lock()
        # Fork-after-init guard: a fork with a thread holding the lock would deadlock
        # the child; _check_pid surfaces InterfaceError instead.
        self._creator_pid = os.getpid()

    def _check_pid(self) -> None:
        if _conn_mod.get_current_pid() != self._creator_pid:
            raise InterfaceError(
                f"MemoryNodeStore used after fork; reconstruct from "
                f"configuration in the target process. (created in "
                f"pid {self._creator_pid}, current pid {_conn_mod.get_current_pid()})"
            )

    def __reduce__(self) -> NoReturn:
        # Pickle/copy would build a FRESH asyncio.Lock on the copy, so two aliases'
        # writers would no longer serialise. Block it (covers pickle/copy/deepcopy).
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — holds an "
            f"asyncio.Lock under a single-owner-per-store discipline "
            f"that does not survive deserialisation; reconstruct from "
            f"the seed addresses in the target process instead."
        )

    async def get_nodes(self) -> Sequence[NodeInfo]:
        self._check_pid()
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        self._check_pid()
        async with self._set_nodes_lock:
            await self._set_nodes_locked(nodes)

    async def _set_nodes_locked(self, nodes: Sequence[NodeInfo]) -> None:
        self._nodes = await _validate_and_normalise_nodes_async(nodes)


# Cooperative-yield cadence for the async per-node validation walk. Local copy of
# cluster._PROBE_TASK_CREATE_YIELD_EVERY (256) to avoid the cluster import cycle.
_NODE_VALIDATE_YIELD_EVERY: Final[int] = 256


def _materialise_and_cap_nodes(
    nodes: Sequence[NodeInfo] | Iterable[NodeInfo],
) -> Sequence[NodeInfo]:
    """Materialise NodeInfo into an indexable sequence and enforce the count cap.

    Generator-tolerant (materialised to a list so len()/indexing work) and shared
    by the sync and async validators so the cap check cannot drift.
    """
    if not isinstance(nodes, (list, tuple)):
        nodes = list(nodes)
    # Cap before per-entry validation so a pathological input can't block the loop.
    # Matches the wire-side _MAX_NODE_COUNT on ServersResponse.
    from dqlitewire.messages.responses import _MAX_NODE_COUNT as _WIRE_MAX_NODES

    if len(nodes) > _WIRE_MAX_NODES:
        raise ValueError(
            f"too many nodes: got {len(nodes)}, max {_WIRE_MAX_NODES} "
            f"(matches wire-side ServersResponse cap)"
        )
    return nodes


def _normalise_one(
    node: NodeInfo,
    seen: set[tuple[str, int]],
    unique: list[NodeInfo],
    parse_address: Callable[[str], tuple[str, int]],
) -> None:
    """Validate / strip / dedup a single NodeInfo into ``unique``.

    Dedups by the parse_address canonical (host, port) so lexical variants (case,
    IPv6 short/long form) collapse to one entry. Shared by sync and async validators.
    """
    if not isinstance(node.address, str):
        raise TypeError(
            f"NodeInfo.address must be 'host:port' string, got {type(node.address).__name__}"
        )
    addr = node.address.strip()
    if not addr:
        raise ValueError("NodeInfo.address must be a non-empty 'host:port' string")
    try:
        canonical = parse_address(addr)
    except ValueError as e:
        raise ValueError(f"NodeInfo.address {addr!r} is not a valid 'host:port': {e}") from e
    if canonical in seen:
        return
    seen.add(canonical)
    if addr is node.address:
        unique.append(node)
    else:
        # Frozen dataclass: rebuild with the stripped address.
        unique.append(NodeInfo(node_id=node.node_id, address=addr, role=node.role))


def _validate_and_normalise_nodes(
    nodes: Sequence[NodeInfo] | Iterable[NodeInfo],
) -> tuple[NodeInfo, ...]:
    """Strip / validate / dedup NodeInfo entries (synchronous).

    Used from sync contexts (MemoryNodeStore.__init__, YamlNodeStore._load_from_disk);
    the async set_nodes paths use _validate_and_normalise_nodes_async.
    """
    capped = _materialise_and_cap_nodes(nodes)
    seen: set[tuple[str, int]] = set()
    unique: list[NodeInfo] = []
    # Local import to avoid the cluster import cycle.
    from dqliteclient.connection import parse_address as _parse_addr_validator

    for node in capped:
        _normalise_one(node, seen, unique, _parse_addr_validator)
    return tuple(unique)


async def _validate_and_normalise_nodes_async(
    nodes: Sequence[NodeInfo] | Iterable[NodeInfo],
) -> tuple[NodeInfo, ...]:
    """Async-yielding twin of _validate_and_normalise_nodes.

    Identical output, but yields every _NODE_VALIDATE_YIELD_EVERY nodes so a large
    input doesn't monopolise the loop. Used by the async set_nodes paths.
    """
    capped = _materialise_and_cap_nodes(nodes)
    seen: set[tuple[str, int]] = set()
    unique: list[NodeInfo] = []
    # Local import to avoid the cluster import cycle (see the sync twin).
    from dqliteclient.connection import parse_address as _parse_addr_validator

    for i, node in enumerate(capped):
        _normalise_one(node, seen, unique, _parse_addr_validator)
        if (i + 1) % _NODE_VALIDATE_YIELD_EVERY == 0:
            await asyncio.sleep(0)
    return tuple(unique)


# Role-string aliases accepted on read (hand-edited files); write() emits the
# canonical integer. Keys are dash/underscore-stripped before lookup.
_YAML_ROLE_STRING_ALIASES: Final[dict[str, NodeRole]] = {
    "voter": NodeRole.VOTER,
    "standby": NodeRole.STANDBY,
    "spare": NodeRole.SPARE,
}

# Cap on the YAML file _load_from_disk will read: bounds a corrupt/co-tenant
# overwrite forcing a huge eager read on the loop thread.
_MAX_YAML_NODE_STORE_BYTES: Final[int] = 1 << 20


class YamlNodeStore(NodeStore):
    """File-backed YAML node store, byte-compatible with go-dqlite (commit d046c95).

    On-disk format matches go-dqlite's NewYamlNodeStore: a list of mappings with
    PascalCase keys and an integer Role (0=Voter, 1=StandBy, 2=Spare)::

        - ID: 1
          Address: node1:9001
          Role: 0

    Lowercased keys or string roles will NOT round-trip against a Go reader, so
    write() always emits this exact shape; reads additionally accept lowercase keys
    and string roles for hand-edited files. Any upstream format change must be
    mirrored here and in the interop test fixture.

    Reads are INTENTIONALLY stricter than go-dqlite: missing ID/Address are rejected
    rather than defaulted (ID=0 is the "no node" sentinel, Address="" is unusable).

    Writes use atomic write-then-rename (tempfile + os.replace) at mode 0600.
    PyYAML is optional: install python-dqlite-client[yaml-store].

    On the exceptional cap-fire RuntimeError exit (see set_nodes), the in-memory
    snapshot becomes eventually-consistent (a deferred disk re-read may overwrite a
    later writer's snapshot); the on-disk state stays correctly serialised. Treat the
    in-memory view as untrusted until set_nodes succeeds again or the store is rebuilt.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Construct a YamlNodeStore rooted at ``path``, eager-loading the file.

        Synchronous: the blocking disk load freezes the loop. Async callers should
        prefer the create() factory, which loads via asyncio.to_thread.
        """
        # Probe PyYAML now so a broken install fails at construction. Do NOT cache the
        # module on self — modules aren't picklable; re-import on demand (cached).
        try:
            import yaml as _yaml  # noqa: F401  - import-time presence check only
        except ImportError as e:
            raise ImportError(
                "YamlNodeStore requires PyYAML; install with "
                "'pip install python-dqlite-client[yaml-store]'"
            ) from e
        self._path = Path(path)
        self._lock = asyncio.Lock()
        # Eager-load so a malformed file raises at construction, not at first get_nodes.
        self._nodes: tuple[NodeInfo, ...] = self._load_from_disk()
        # Fork-after-init guard: the child must reconstruct (own lock, re-read file).
        self._creator_pid = os.getpid()

    @classmethod
    async def create(cls, path: str | os.PathLike[str]) -> "YamlNodeStore":
        """Async-safe constructor: load via asyncio.to_thread so the loop stays free."""
        try:
            import yaml as _yaml  # noqa: F401  - import-time presence check only
        except ImportError as e:
            raise ImportError(
                "YamlNodeStore requires PyYAML; install with "
                "'pip install python-dqlite-client[yaml-store]'"
            ) from e
        store = cls.__new__(cls)
        store._path = Path(path)
        store._lock = asyncio.Lock()
        store._nodes = await asyncio.to_thread(store._load_from_disk)
        store._creator_pid = os.getpid()
        return store

    def _check_pid(self) -> None:
        if _conn_mod.get_current_pid() != self._creator_pid:
            raise InterfaceError(
                f"YamlNodeStore used after fork; reconstruct from "
                f"configuration in the target process. (created in "
                f"pid {self._creator_pid}, current pid {_conn_mod.get_current_pid()})"
            )

    @property
    def path(self) -> Path:
        return self._path

    def _load_from_disk(self) -> tuple[NodeInfo, ...]:
        # lstat (no symlink follow) so a dangling symlink surfaces as an OSError via
        # O_NOFOLLOW below rather than as a silent "missing file". Only
        # FileNotFoundError collapses to the empty-store fallback.
        try:
            self._path.lstat()
        except FileNotFoundError:
            return ()
        # Open-once + fstat + bounded read closes the stat/read TOCTOU window
        # (CWE-367) and O_NOFOLLOW rejects a symlink swap (CWE-59); O_CLOEXEC
        # avoids fd leaks across exec.
        try:
            fd = os.open(
                str(self._path),
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as e:
            raise ClusterError(f"YamlNodeStore: cannot open {self._path}: {e}") from e
        try:
            try:
                fstat = os.fstat(fd)
            except OSError as e:
                raise ClusterError(f"YamlNodeStore: cannot fstat {self._path}: {e}") from e
            if fstat.st_size > _MAX_YAML_NODE_STORE_BYTES:
                raise ClusterError(
                    f"YamlNodeStore: {self._path} exceeds maximum size "
                    f"({fstat.st_size} > {_MAX_YAML_NODE_STORE_BYTES} bytes)"
                )
            # cap+1 so a file that grew after fstat still trips the cap below.
            try:
                data = os.read(fd, _MAX_YAML_NODE_STORE_BYTES + 1)
            except OSError as e:
                raise ClusterError(f"YamlNodeStore: cannot read {self._path}: {e}") from e
        finally:
            os.close(fd)
        if len(data) > _MAX_YAML_NODE_STORE_BYTES:
            raise ClusterError(
                f"YamlNodeStore: {self._path} grew past the maximum "
                f"size ({_MAX_YAML_NODE_STORE_BYTES} bytes) during the "
                f"read"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ClusterError(f"YamlNodeStore: {self._path} contains non-UTF-8 bytes: {e}") from e
        if not text.strip():
            return ()
        import yaml

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ClusterError(
                f"YamlNodeStore: malformed YAML in {self._path}: {_cap_and_sanitize(e)}"
            ) from e
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ClusterError(
                f"YamlNodeStore: {self._path} top-level must be a "
                f"YAML list, got {type(raw).__name__}"
            )
        result: list[NodeInfo] = []
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] must be a "
                    f"mapping, got {type(entry).__name__}"
                )
            # Accept PascalCase and lowercase keys. Explicit ``is None`` fallback
            # (not get(k, get(alias))) so an explicit ``ID: null`` still falls
            # through to the lowercase alias.
            node_id_raw = entry.get("ID")
            if node_id_raw is None:
                node_id_raw = entry.get("id")
            address_raw = entry.get("Address")
            if address_raw is None:
                address_raw = entry.get("address")
            role_raw = entry.get("Role")
            if role_raw is None:
                role_raw = entry.get("role")
            if node_id_raw is None:
                raise ClusterError(f"YamlNodeStore: {self._path}[{idx}] missing 'ID'")
            if address_raw is None:
                raise ClusterError(f"YamlNodeStore: {self._path}[{idx}] missing 'Address'")
            # Reject bool/float before int(): int(True)/int(3.7) would silently
            # coerce. Bare YAML ints are already Python int; only a string "5" falls
            # through to the int() arm.
            if isinstance(node_id_raw, (bool, float)):
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] 'ID' must be integer "
                    f"(not float / bool), got {_cap_and_sanitize(repr(node_id_raw))}"
                )
            if not isinstance(node_id_raw, int):
                try:
                    node_id = int(node_id_raw)
                except (TypeError, ValueError) as e:
                    raise ClusterError(
                        f"YamlNodeStore: {self._path}[{idx}] 'ID' must be "
                        f"integer, got {_cap_and_sanitize(repr(node_id_raw))}"
                    ) from e
            else:
                node_id = node_id_raw
            # Reject non-str here so the diagnostic blames the YAML schema rather than
            # a downstream parse_address failure on a coerced value.
            if not isinstance(address_raw, str):
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] 'Address' must be "
                    f"str, got {type(address_raw).__name__}"
                )
            address = address_raw
            # Role: integer (go-dqlite) or string alias; missing -> Voter.
            if role_raw is None:
                role = NodeRole.VOTER
            elif isinstance(role_raw, bool):
                # bool passes isinstance(_, int); reject before the integer arm.
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] 'Role' must be int or str, got bool"
                )
            elif isinstance(role_raw, int):
                try:
                    role = NodeRole(role_raw)
                except ValueError as e:
                    raise ClusterError(
                        f"YamlNodeStore: {self._path}[{idx}] 'Role' "
                        f"integer {_cap_and_sanitize(role_raw)} is not a valid NodeRole"
                    ) from e
            elif isinstance(role_raw, str):
                normalised = role_raw.strip().lower().replace("-", "").replace("_", "")
                if normalised not in _YAML_ROLE_STRING_ALIASES:
                    raise ClusterError(
                        f"YamlNodeStore: {self._path}[{idx}] 'Role' "
                        f"string {_cap_and_sanitize(repr(role_raw))} "
                        f"not one of voter/stand-by/spare"
                    )
                role = _YAML_ROLE_STRING_ALIASES[normalised]
            else:
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] 'Role' must be "
                    f"int or str, got {type(role_raw).__name__}"
                )
            result.append(NodeInfo(node_id=node_id, address=address, role=role))
        # Canonicalise on load through the same pipeline set_nodes enforces on write,
        # so reads accept exactly what writes produce.
        try:
            return _validate_and_normalise_nodes(result)
        except (TypeError, ValueError) as e:
            raise ClusterError(f"YamlNodeStore: {self._path}: {_cap_and_sanitize(e)}") from e

    async def get_nodes(self) -> Sequence[NodeInfo]:
        self._check_pid()
        return self._nodes

    async def _write_and_publish(
        self,
        normalised: tuple[NodeInfo, ...],
        payload: list[dict[str, Any]],
    ) -> None:
        """Commit ``payload`` to disk (worker thread) then publish ``normalised``.

        Shielded by the caller so a cancel can't land between the disk write and the
        in-memory assignment and leave disk/self._nodes divergent.
        """
        await asyncio.to_thread(self._serialise_and_write_sync, payload)
        self._nodes = normalised

    def _serialise_and_write_sync(self, payload: list[dict[str, Any]]) -> None:
        """Serialise ``payload`` to YAML and write it atomically (worker-thread body)."""
        import yaml

        text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        self._write_atomic_sync(text)

    def _write_atomic_sync(self, text: str) -> None:
        """Synchronous atomic write-then-rename ritual (runs on a worker thread)."""
        parent = self._path.parent if str(self._path.parent) else Path(".")
        fd_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(text)
                tmp.flush()
                os.fsync(tmp.fileno())
                fd_path = tmp.name
            # mkstemp creates the file 0600 on POSIX (matches go-dqlite). No chmod:
            # it would open a TOCTOU window. Debug assertion pins the invariant.
            if __debug__ and os.name == "posix":
                actual_mode = stat.S_IMODE(os.stat(fd_path).st_mode)
                assert actual_mode == 0o600, (
                    f"tempfile created with unexpected mode {oct(actual_mode)}; "
                    "set_nodes pre-rename mode discipline is broken"
                )
            os.replace(fd_path, self._path)
            fd_path = None  # ownership transferred
            # Fsync the parent dir so the rename survives power loss (POSIX rename is
            # atomic for visibility but the dir entry can revert). Best-effort:
            # suppress OSError where dir fsync is unsupported (Windows, some FUSE).
            with contextlib.suppress(OSError):
                dir_fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            if fd_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(fd_path)

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        self._check_pid()
        # Manual acquire/release (not ``async with``) so the lock is held across the
        # shielded write even when an outer cancel lands mid-to_thread: ``async with``
        # would release in __aexit__ while the shielded write still ran, letting a
        # concurrent writer race the orphan and reintroduce disk/memory divergence.
        await self._lock.acquire()
        try:
            # Validate before writing so set_nodes never persists bytes that
            # _load_from_disk would later reject. Inside the lock so two writers
            # can't race the pre-pass.
            normalised = await _validate_and_normalise_nodes_async(nodes)
            payload = [
                {
                    "ID": int(n.node_id),
                    "Address": str(n.address),
                    "Role": int(n.role),  # integer enum value (go-dqlite shape)
                }
                for n in normalised
            ]
            # Offload serialisation + I/O to a worker thread so a slow fsync doesn't
            # freeze the loop. Shield the to_thread await and the in-memory assignment
            # together so a cancel can't land between the disk write and self._nodes
            # update and leave them divergent.
            inner = asyncio.ensure_future(self._write_and_publish(normalised, payload))
            try:
                await asyncio.shield(inner)
            except asyncio.CancelledError:
                # Outer await cancelled but the shielded inner task runs on. Drain it
                # under the lock (looping on repeated cancels) so a concurrent writer
                # can't race the orphan. Bounded by _MAX_CANCEL_DRAIN_ITERS so a wedged
                # worker can't spin here forever holding the lock process-wide.
                cancel_count = 0
                while not inner.done():
                    try:
                        await asyncio.shield(inner)
                    except asyncio.CancelledError:
                        cancel_count += 1
                        if cancel_count > _MAX_CANCEL_DRAIN_ITERS:
                            logger.warning(
                                "set_nodes: cancel-drain budget exceeded "
                                "(%d cancels absorbed); worker thread "
                                "appears stuck. Releasing lock and "
                                "surfacing RuntimeError.",
                                cancel_count,
                            )
                            # Cancel inner and give one bounded settle round;
                            # wait_for's timeout keeps the cap honest against a wedged
                            # worker, suppress ensures we reach the raise below.
                            inner.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await asyncio.wait_for(asyncio.shield(inner), timeout=0.5)
                            # Schedule (don't await — this task is cancelling) a
                            # disk re-read to reconcile self._nodes after the lock is
                            # released; eventual consistency is the best available once
                            # we give up. _observe_drain_exception swallows any raise so
                            # a failed re-read isn't an unretrieved-task warning.
                            from dqliteclient.cluster import (
                                _observe_drain_exception,
                            )

                            def _reconcile_in_memory() -> None:
                                try:
                                    self._nodes = self._load_from_disk()
                                except Exception:
                                    logger.warning(
                                        "set_nodes cap-fire: "
                                        "_load_from_disk also failed; "
                                        "in-memory snapshot may lag "
                                        "the actual on-disk state "
                                        "until the next successful "
                                        "set_nodes",
                                        exc_info=True,
                                    )

                            reload_task = asyncio.ensure_future(
                                asyncio.to_thread(_reconcile_in_memory)
                            )
                            reload_task.add_done_callback(_observe_drain_exception)
                            raise RuntimeError(
                                "YamlNodeStore.set_nodes: cancel-drain "
                                f"budget exceeded after {cancel_count} "
                                "cancels absorbed; worker thread is "
                                "stuck on fsync."
                            ) from None
                        continue
                    except Exception:
                        # Non-cancel inner failure (disk error): let it escape so the
                        # caller sees the real cause, not a generic CancelledError.
                        logger.debug(
                            "set_nodes: inner write raised during cancel drain",
                            exc_info=True,
                        )
                        raise
                # If inner finished with an exception the drain loop never observed,
                # read it now to mark the task retrieved (avoids the asyncio
                # "exception never retrieved" warning); the CancelledError still
                # re-raises below.
                if inner.exception() is not None:
                    logger.debug(
                        "set_nodes: inner write raised before cancel drain entry",
                        exc_info=inner.exception(),
                    )
                raise
        finally:
            self._lock.release()
