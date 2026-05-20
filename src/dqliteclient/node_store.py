"""Node store interfaces for cluster discovery."""

import asyncio
import contextlib
import logging
import os
import stat
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from dqliteclient import connection as _conn_mod
from dqliteclient.exceptions import ClusterError, InterfaceError
from dqlitewire import NodeRole

__all__ = ["MemoryNodeStore", "NodeInfo", "NodeStore", "YamlNodeStore"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a cluster node.

    Frozen so callers holding a reference from ``get_nodes()`` can't
    accidentally mutate store state (and so instances are hashable
    and usable as set/dict keys for deduplication).
    """

    node_id: int
    """Dqlite raft node id. Typically 1-based, assigned by the
    cluster when the node joined. In the
    ``MemoryNodeStore.__init__(addresses=[...])`` seeding path the id
    is synthesised (``i + 1``) and does NOT correspond to the
    cluster-side id — it's a placeholder used for ordering and
    dedup. Callers that need the authoritative id should populate
    the store from a control-plane source that knows it."""

    address: str
    """``host:port`` string. IPv6 addresses must be bracketed
    (``[::1]:9000``). Parsed by
    ``dqliteclient.parse_address`` — hostnames are ASCII-only and
    lowercased, IP literals are canonicalised."""

    role: NodeRole
    """Raft role: ``VOTER``, ``STANDBY``, or ``SPARE``. Standby and
    spare nodes cannot become leader —
    ``ClusterClient.find_leader`` sorts voters first before probing."""

    def __post_init__(self) -> None:
        # Validate role at construction time so a caller-built
        # ``NodeInfo`` carrying a bogus role can't reach the wire-
        # encoder and silently emit an unrecognised value onto the
        # wire. Mirrors the wire-side
        # ``dqlitewire.messages.responses.NodeInfo.__post_init__``.
        # ``IntEnum`` accepts ``NodeRole(0)`` etc. but not
        # ``NodeRole(999)``, so coerce bare ints into the enum and let
        # ``ValueError`` from the constructor surface as a constructor-
        # time error rather than at peer-decode time.
        if not isinstance(self.role, NodeRole):
            try:
                coerced = NodeRole(self.role)
            except ValueError as e:
                raise ValueError(
                    f"NodeInfo: unknown role {self.role!r}; valid roles are "
                    f"0 (VOTER), 1 (STANDBY), 2 (SPARE)"
                ) from e
            object.__setattr__(self, "role", coerced)


@runtime_checkable
class NodeStore(Protocol):
    """Structural interface for storing cluster node information.

    PEP 544 ``Protocol`` so third-party stores need only implement
    ``get_nodes`` / ``set_nodes`` without inheriting from this class.
    No callers in this codebase use ``isinstance(x, NodeStore)`` —
    the contract is structural — and ``MemoryNodeStore`` is the only
    in-tree implementation. ``@runtime_checkable`` is set so a future
    caller that wants ``isinstance`` introspection still gets it.
    """

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Return an immutable snapshot of known nodes.

        Implementations MUST return a Sequence that will not be mutated
        after return. A ``tuple`` is preferred (``MemoryNodeStore``
        uses one); a freshly-copied ``list`` is also acceptable as long
        as the implementation guarantees no subsequent mutation. Callers
        — notably ``ClusterClient.find_leader`` — iterate the returned
        value without defensive copies, and iteration is interleaved
        with ``await`` points. Returning a live backing collection that
        mutates under ``set_nodes()`` would produce torn reads during
        iteration.
        """
        ...

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes.

        Implementations MUST publish the update atomically from the
        perspective of ``get_nodes()``: a concurrent reader must see
        either the old snapshot or the new one, never a partially
        constructed sequence. ``MemoryNodeStore`` achieves this with a
        tuple-reference swap.
        """
        ...


class MemoryNodeStore(NodeStore):
    """In-memory node store.

    Backs storage with an immutable tuple so readers can safely be
    handed a direct reference without defensive copies and so a
    concurrent ``set_nodes()`` never produces a torn view.

    Concurrent writers are serialised by an ``asyncio.Lock``: two
    callers each invoking ``set_nodes`` no longer race on the final
    tuple assignment (last-writer-wins lost-update). Mirrors the
    discipline ``YamlNodeStore`` already applies for the same
    reason.
    """

    def __init__(
        self,
        addresses: Sequence[str] | None = None,
        *,
        initial_addresses: Sequence[str] | None = None,
    ) -> None:
        """Seed with a list of ``host:port`` addresses.

        ``addresses`` is the preferred name (matches the sibling
        ``ClusterClient.from_addresses`` / ``ConnectionPool`` /
        top-level ``connect`` APIs). ``initial_addresses`` is a
        deprecated alias kept for back-compat; prefer ``addresses`` in
        new code.
        """
        if addresses is not None and initial_addresses is not None:
            raise TypeError("Pass only one of 'addresses' or 'initial_addresses'")
        if initial_addresses is not None and addresses is None:
            # Surface the deprecation tag in the docstring as a runtime
            # signal so operators can grep their test suites under
            # ``-W error::DeprecationWarning`` and find every call site
            # to migrate. ``stacklevel=2`` directs the warning at the
            # caller's invocation, not at this line in __init__ — the
            # standard ``warnings.warn`` discipline for deprecated
            # constructor kwargs.
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
            # Build raw NodeInfo entries (synthetic id=i+1, role=VOTER,
            # address-as-string-from-input) and route through the shared
            # ``_validate_and_normalise_nodes`` helper so the strip /
            # parse / dedup pipeline lives in exactly one place. Without
            # this, the per-string validation loop here and the
            # per-NodeInfo loop in the helper drifted on input shape
            # (``Sequence[str]`` vs ``Sequence[NodeInfo]``); future
            # rule additions had to land in two places.
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
        # Lock the ``set_nodes`` critical section. Constructed eagerly
        # in ``__init__`` — `asyncio.Lock` is loop-agnostic at
        # construction time on Python 3.10+ (binds to the loop on
        # first ``acquire()``). The previous lazy ``None``-then-
        # initialise shape lost the mutual-exclusion contract under
        # concurrent first-time callers: both observed ``None`` on the
        # same scheduling slice, each constructed a fresh ``Lock``,
        # and the second STORE_ATTR won — both proceeded to
        # ``_set_nodes_locked`` in parallel and one update was lost.
        # Mirrors ``YamlNodeStore.__init__`` which has always done
        # eager init.
        self._set_nodes_lock: asyncio.Lock = asyncio.Lock()
        # Fork-after-init guard. The lock is loop-agnostic on the
        # supported Python versions, but a multi-threaded parent
        # forking with one thread inside ``set_nodes`` would leave
        # the child with a permanently-held lock (first child
        # ``acquire()`` blocks forever). Surface the canonical
        # ``InterfaceError`` instead of either hanging or raising
        # an opaque asyncio diagnostic. Symmetric with
        # :class:`ClusterClient` / :class:`DqliteConnection` /
        # :class:`ConnectionPool`.
        self._creator_pid = os.getpid()

    def _check_pid(self) -> None:
        if _conn_mod._current_pid != self._creator_pid:
            raise InterfaceError(
                f"MemoryNodeStore used after fork; reconstruct from "
                f"configuration in the target process. (created in "
                f"pid {self._creator_pid}, current pid {_conn_mod.get_current_pid()})"
            )

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Get list of known nodes."""
        self._check_pid()
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes.

        Mirrors the strip / dedup / empty-rejection validation done
        in ``__init__``. Without this, a runtime update with a
        whitespace-laden or duplicated address would leak through
        and surface deep inside ``find_leader`` (whitespace) or
        inflate the probe count and per-node error lines
        (duplicates).

        Concurrent callers are serialised via an asyncio.Lock: two
        ``set_nodes`` invocations race-free on the final tuple
        assignment so neither caller's update is silently lost.
        """
        self._check_pid()
        async with self._set_nodes_lock:
            await self._set_nodes_locked(nodes)

    async def _set_nodes_locked(self, nodes: Sequence[NodeInfo]) -> None:
        self._nodes = _validate_and_normalise_nodes(nodes)


def _validate_and_normalise_nodes(nodes: Sequence[NodeInfo]) -> tuple[NodeInfo, ...]:
    """Strip / validate / dedup ``NodeInfo`` entries.

    Shared by every node-store entry point that constructs an
    in-memory node tuple from raw input — :meth:`MemoryNodeStore.__init__`
    (after wrapping each seed string in a synthetic ``NodeInfo``),
    :meth:`MemoryNodeStore._set_nodes_locked`, :meth:`YamlNodeStore.set_nodes`,
    and :meth:`YamlNodeStore._load_from_disk`. The shared helper
    eliminates rule-drift across the four call sites: a future
    address-validation rule (e.g. "reject IPv6 link-local") lands in
    one place.

    Validation pipeline:

    1. Reject non-string ``NodeInfo.address`` with ``TypeError``.
    2. Strip leading/trailing whitespace (operator-friendly canonicalisation).
    3. Reject empty stripped addresses with ``ValueError``.
    4. Validate ``host:port`` shape via ``parse_address``.
    5. Deduplicate by the ``parse_address`` canonical ``(host, port)``
       tuple so address variants that differ only in lexical shape
       (mixed-case hostnames per RFC 1035, IPv6 short-vs-long form,
       non-canonical IPv4 literals) dedup to a single entry.
       Mirrors the canonical-tuple discipline used by
       :func:`dqliteclient.cluster._addr_equiv`.
    6. Return frozen ``NodeInfo`` tuples with the stripped address so
       downstream lookups by address match.
    """
    # Length-cap before per-entry validation so a pathological 1 M-entry
    # input from app-side misuse cannot block the event loop on
    # parse_address loops under the asyncio.Lock held by
    # YamlNodeStore.set_nodes. Matches the wire-side cap
    # _MAX_NODE_COUNT applied to ServersResponse.
    from dqlitewire.messages.responses import _MAX_NODE_COUNT as _WIRE_MAX_NODES

    if len(nodes) > _WIRE_MAX_NODES:
        raise ValueError(
            f"too many nodes: got {len(nodes)}, max {_WIRE_MAX_NODES} "
            f"(matches wire-side ServersResponse cap)"
        )
    seen: set[tuple[str, int]] = set()
    unique: list[NodeInfo] = []
    # Local import to avoid the cluster import cycle (cluster imports
    # node_store; node_store would otherwise need cluster's
    # ``connection`` module at module-import time).
    from dqliteclient.connection import parse_address as _parse_addr_validator

    for node in nodes:
        if not isinstance(node.address, str):
            raise TypeError(
                f"NodeInfo.address must be 'host:port' string, got {type(node.address).__name__}"
            )
        addr = node.address.strip()
        if not addr:
            raise ValueError("NodeInfo.address must be a non-empty 'host:port' string")
        try:
            canonical = _parse_addr_validator(addr)
        except ValueError as e:
            raise ValueError(f"NodeInfo.address {addr!r} is not a valid 'host:port': {e}") from e
        if canonical in seen:
            continue
        seen.add(canonical)
        if addr is node.address:
            unique.append(node)
        else:
            # NodeInfo is a frozen dataclass; rebuild with the stripped
            # address so a downstream lookup by-address matches.
            unique.append(NodeInfo(node_id=node.node_id, address=addr, role=node.role))
    return tuple(unique)


# Role-string aliases accepted on read for ergonomics with hand-edited
# files. ``write()`` always emits the canonical integer Role enum
# value, matching go-dqlite's ``yaml.v2`` output of ``NodeRole int``.
# Normalisation strips dashes and underscores so ``stand-by``,
# ``stand_by``, and ``standby`` all map to the same role.
_YAML_ROLE_STRING_ALIASES: Final[dict[str, NodeRole]] = {
    "voter": NodeRole.VOTER,
    "standby": NodeRole.STANDBY,
    "spare": NodeRole.SPARE,
}

# Cap the size of the YAML node-store file accepted by _load_from_disk.
# A legitimate dqlite node store is a handful of entries (a few KB);
# 1 MiB is generous bounding while preventing a corrupt / co-tenant
# overwrite from forcing a multi-MB eager read at construction (which
# runs synchronously on the event-loop thread).
_MAX_YAML_NODE_STORE_BYTES: Final[int] = 1 << 20


class YamlNodeStore(NodeStore):
    """File-backed node store using YAML, byte-compatible with go-dqlite.

    On-disk format matches go-dqlite's ``NewYamlNodeStore`` (verified
    against ``client/store.go`` + ``internal/protocol/store.go`` at
    commit ``d046c95``). The ``NodeInfo`` struct in Go has yaml tags
    ``ID``, ``Address``, ``Role`` (PascalCase, NOT lowercase) and
    ``Role`` is a ``NodeRole int`` with no custom MarshalYAML, so
    ``yaml.v2`` serialises it as the integer enum value:

    .. code-block:: yaml

        - ID: 1
          Address: node1:9001
          Role: 0
        - ID: 2
          Address: node2:9002
          Role: 1

    where Role is 0=Voter, 1=StandBy, 2=Spare.

    A mixed-language deployment (Go service + Python service sharing
    one bootstrap file) requires this exact shape. Lowercased keys
    or string roles will NOT round-trip against a Go reader.

    On read we accept lowercase aliases (``id``, ``address``, ``role``)
    and string role values for ergonomics with hand-edited files,
    but ``write()`` always emits the canonical PascalCase + integer
    form.

    Writes go through atomic write-then-rename via
    :func:`tempfile.NamedTemporaryFile` + :func:`os.replace`, with
    file mode 0600 to match go-dqlite's ``renameio.WriteFile(...,
    0600)``. Same-directory rename is atomic on POSIX.

    Format pinning: this implementation tracks go-dqlite ``d046c95``
    for the on-disk YAML layout (key names, ordering, file mode). Any
    future upstream format change must be reflected here AND in the
    interop test fixture.

    Reader-side strictness is INTENTIONALLY stricter than go-dqlite:
    where go-dqlite's ``yaml.Unmarshal`` defaults missing ``ID`` /
    ``Address`` fields to zero / empty string, this loader rejects
    them with ``ClusterError``. ``ID=0`` is the upstream "no node"
    sentinel and ``Address=""`` is unusable, so silently accepting
    degenerate entries would corrupt the in-memory store with rows
    the downstream ``_validate_and_normalise_nodes`` would reject
    anyway with a less specific error site. The parity claim above is
    about the *write* shape, not reader tolerance.

    PyYAML is an optional dependency: install
    ``python-dqlite-client[yaml-store]`` to use this class. The base
    install does not pull PyYAML.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        # Verify PyYAML is importable so a malformed install raises
        # at construction (operator-facing) rather than at first
        # ``get_nodes``. Do NOT cache the module reference on
        # ``self`` — module objects are not picklable, and caching
        # one would break ``pickle.dumps(store)`` with a confusing
        # generic ``TypeError: cannot pickle 'module' object``
        # (no class name, no remediation hint). Re-import on demand
        # in ``_load_from_disk`` / ``_save_to_disk``; the import is
        # cached after first use (~µs cost per call site).
        try:
            import yaml as _yaml  # noqa: F401  - import-time presence check only
        except ImportError as e:
            raise ImportError(
                "YamlNodeStore requires PyYAML; install with "
                "'pip install python-dqlite-client[yaml-store]'"
            ) from e
        self._path = Path(path)
        self._lock = asyncio.Lock()
        # Eager-load and validate so a malformed file raises at
        # construction (operator-facing) rather than at first
        # ``get_nodes`` (deep inside ``find_leader``).
        self._nodes: tuple[NodeInfo, ...] = self._load_from_disk()
        # Fork-after-init guard. The on-disk state survives fork
        # as plain data, but the asyncio.Lock and the in-memory
        # tuple are parent-process state. The child must
        # reconstruct from the configuration in the target process
        # so it re-reads the YAML file and binds its own lock.
        self._creator_pid = os.getpid()

    def _check_pid(self) -> None:
        if _conn_mod._current_pid != self._creator_pid:
            raise InterfaceError(
                f"YamlNodeStore used after fork; reconstruct from "
                f"configuration in the target process. (created in "
                f"pid {self._creator_pid}, current pid {_conn_mod.get_current_pid()})"
            )

    @property
    def path(self) -> Path:
        """Path of the backing YAML file (read-only accessor)."""
        return self._path

    def _load_from_disk(self) -> tuple[NodeInfo, ...]:
        # Three distinct paths:
        #   1. Missing file -> empty tuple (matches go-dqlite which
        #      tolerates a missing file at NewYamlNodeStore time).
        #   2. Empty / whitespace-only file -> empty tuple.
        #   3. Corrupt YAML -> ClusterError with file path.
        if not self._path.exists():
            return ()
        # Stat-precheck the file size before reading: a pathologically
        # large file (corrupt YAML / co-tenant write / accidental
        # overwrite) would otherwise be fully loaded into memory by
        # ``read_text`` before yaml.safe_load even sees it. 1 MiB is
        # generous for any legitimate dqlite node store (a real store
        # is a few KB) while bounding the worst-case eager-load cost
        # at construction time (this runs synchronously on the
        # event-loop thread before any to_thread dispatch).
        try:
            stat = self._path.stat()
        except OSError as e:
            raise ClusterError(f"YamlNodeStore: cannot stat {self._path}: {e}") from e
        if stat.st_size > _MAX_YAML_NODE_STORE_BYTES:
            raise ClusterError(
                f"YamlNodeStore: {self._path} exceeds maximum size "
                f"({stat.st_size} > {_MAX_YAML_NODE_STORE_BYTES} bytes)"
            )
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as e:
            raise ClusterError(f"YamlNodeStore: cannot read {self._path}: {e}") from e
        if not text.strip():
            return ()
        import yaml

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ClusterError(f"YamlNodeStore: malformed YAML in {self._path}: {e}") from e
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
            # Accept PascalCase (canonical / go-dqlite) and lowercase
            # (hand-edited) keys.
            node_id_raw = entry.get("ID", entry.get("id"))
            address_raw = entry.get("Address", entry.get("address"))
            role_raw = entry.get("Role", entry.get("role"))
            if node_id_raw is None:
                raise ClusterError(f"YamlNodeStore: {self._path}[{idx}] missing 'ID'")
            if address_raw is None:
                raise ClusterError(f"YamlNodeStore: {self._path}[{idx}] missing 'Address'")
            # Reject bool BEFORE int(): bool is an int subclass in
            # Python and ``int(True) == 1`` would silently coerce a
            # ``True``-typed YAML value into node_id 1.
            if isinstance(node_id_raw, bool) or not isinstance(node_id_raw, int):
                try:
                    node_id = int(node_id_raw)
                except (TypeError, ValueError) as e:
                    raise ClusterError(
                        f"YamlNodeStore: {self._path}[{idx}] 'ID' must be "
                        f"integer, got {node_id_raw!r}"
                    ) from e
            else:
                node_id = node_id_raw
            address = str(address_raw)
            # Role: integer canonical (go-dqlite) OR string alias
            # ("voter"/"stand-by"/"spare") for hand-edited files.
            # Missing -> default Voter (matches the seed-list path
            # in MemoryNodeStore.__init__ which assumes VOTER).
            if role_raw is None:
                role = NodeRole.VOTER
            elif isinstance(role_raw, bool):
                # bool slips through ``isinstance(_, int)`` — reject
                # before the integer arm to avoid silent coercion.
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] 'Role' must be int or str, got bool"
                )
            elif isinstance(role_raw, int):
                try:
                    role = NodeRole(role_raw)
                except ValueError as e:
                    raise ClusterError(
                        f"YamlNodeStore: {self._path}[{idx}] 'Role' "
                        f"integer {role_raw} is not a valid NodeRole"
                    ) from e
            elif isinstance(role_raw, str):
                # go-dqlite's ``String()`` emits "voter"/"stand-by"/"spare".
                normalised = role_raw.strip().lower().replace("-", "").replace("_", "")
                if normalised not in _YAML_ROLE_STRING_ALIASES:
                    raise ClusterError(
                        f"YamlNodeStore: {self._path}[{idx}] 'Role' "
                        f"string {role_raw!r} not one of voter/stand-by/spare"
                    )
                role = _YAML_ROLE_STRING_ALIASES[normalised]
            else:
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] 'Role' must be "
                    f"int or str, got {type(role_raw).__name__}"
                )
            result.append(NodeInfo(node_id=node_id, address=address, role=role))
        # Run the parsed entries through the shared strip / dedup /
        # parse_address pipeline so a hand-edited YAML file with
        # whitespace-laden or duplicated addresses canonicalises at
        # load time (matching the discipline ``set_nodes`` enforces on
        # write). Without this, the load path accepted what
        # ``set_nodes`` would now reject — read/write asymmetry that
        # left the docstring's "matches set_nodes" claim aspirational.
        # ``_validate_and_normalise_nodes`` raises ``ValueError`` on a
        # bad shape (matching the existing fail-fast posture below)
        # and returns the canonical tuple on success.
        try:
            return _validate_and_normalise_nodes(result)
        except (TypeError, ValueError) as e:
            raise ClusterError(f"YamlNodeStore: {self._path}: {e}") from e

    async def get_nodes(self) -> Sequence[NodeInfo]:
        self._check_pid()
        return self._nodes

    async def _write_and_publish(
        self,
        normalised: tuple[NodeInfo, ...],
        text: str,
    ) -> None:
        """Atomically commit ``text`` to disk and publish the
        ``normalised`` membership in memory.

        Called by :meth:`set_nodes` wrapped in :func:`asyncio.shield`
        so a cancel arriving between the worker-thread return and
        the in-memory assignment cannot leave disk and ``self._nodes``
        divergent — see the long-form note at the call site.
        """
        await asyncio.to_thread(self._write_atomic_sync, text)
        # Store the canonical (validated, deduped) tuple in memory so
        # get_nodes returns the same shape that _load_from_disk would
        # surface after restart.
        self._nodes = normalised

    def _write_atomic_sync(self, text: str) -> None:
        """Synchronous body of the atomic-rename ritual.

        Extracted so :meth:`set_nodes` can dispatch via
        :func:`asyncio.to_thread` and keep the event loop responsive
        — ``yaml.safe_dump`` is cheap CPU, but ``os.fsync`` against
        a contended disk (slow NAS, encrypted volume, full-disk SSD
        GC pause) can stall for tens to hundreds of milliseconds and
        the directory fsync stalls again. Every coroutine on the
        loop — Connection reader tasks, pool acquirers parked on
        ``_pool.get``, heartbeat probes, SQLAlchemy ``do_ping`` — is
        frozen for the duration without the thread dispatch.
        """
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
            # ``tempfile.NamedTemporaryFile`` opens via
            # ``mkstemp`` which creates the file with mode 0600 on
            # POSIX (matching go-dqlite's ``renameio.WriteFile``
            # 0600 discipline). No additional chmod is needed —
            # adding one creates a TOCTOU window where bytes are
            # already on disk before any chmod tightening could
            # take effect, if a future tempfile default ever
            # relaxed to 0644. Debug-mode assertion pins the
            # invariant so a future regression in tempfile defaults
            # is caught at test time.
            if __debug__ and os.name == "posix":
                actual_mode = stat.S_IMODE(os.stat(fd_path).st_mode)
                assert actual_mode == 0o600, (
                    f"tempfile created with unexpected mode {oct(actual_mode)}; "
                    "set_nodes pre-rename mode discipline is broken"
                )
            os.replace(fd_path, self._path)
            fd_path = None  # ownership transferred
            # Sync the parent directory so the rename's metadata
            # block is durable across a hard reboot. POSIX rename
            # is atomic for visibility but the directory entry
            # change can revert after a power loss / kernel
            # panic / VM hard-stop without the directory fsync.
            # More durable than go-dqlite's ``renameio.WriteFile``,
            # which fsyncs the file but not the parent directory
            # (per the upstream package: "concerns itself only
            # with atomicity"). Mirrors the standard atomic-rename
            # + parent-fsync pattern (git, sqlite WAL, etc.).
            # Best-effort: non-POSIX (Windows) and some FUSE
            # filesystems don't support fsync on directories —
            # suppress OSError so the rename is still visible
            # even if the durability barrier isn't available.
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
        # Serialise concurrent writes — file rename is atomic but two
        # writers racing the load -> serialise -> write sequence
        # would lose one update. asyncio.Lock keeps the in-process
        # last-writer-wins ordering deterministic.
        import yaml

        # Manual ``acquire`` / ``release`` (rather than ``async with``)
        # so the lock is held across the shielded inner write even
        # when an outer cancel lands mid-``to_thread``. The previous
        # ``async with self._lock: await asyncio.shield(...)`` shape
        # released the lock in the ``__aexit__`` while the shielded
        # ``_write_and_publish`` was still running in the background
        # — a concurrent ``set_nodes`` could then acquire the freed
        # lock and race the orphan shielded write, reintroducing the
        # disk/memory divergence the prior shield-only fix was
        # supposed to prevent. Capturing the inner task and awaiting it under
        # the lock on the cancel path keeps the second writer parked
        # until the first writer's commit truly finishes. Go's
        # ``store.go::SetServers`` runs under a ``sync.Mutex`` which
        # is uninterruptible for the same reason.
        await self._lock.acquire()
        try:
            # Run the same strip / dedup / parse_address validation
            # pipeline as MemoryNodeStore. Without this, set_nodes
            # could persist whitespace-laden / duplicated / malformed
            # entries that the same _load_from_disk loader would later
            # refuse — postcondition violation (set_nodes succeeded but
            # the next process startup raises ValueError on the very
            # bytes set_nodes wrote). Validation is INSIDE the lock so
            # two concurrent set_nodes cannot race the validation
            # pre-pass and clobber each other's writes.
            normalised = _validate_and_normalise_nodes(nodes)
            payload = [
                {
                    "ID": int(n.node_id),
                    "Address": str(n.address),
                    "Role": int(n.role),  # integer enum value (go-dqlite shape)
                }
                for n in normalised
            ]
            text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
            # Dispatch the sync I/O ritual (yaml.safe_dump's tempfile
            # write, fsync, rename, parent-dir fsync, unlink cleanup)
            # to a worker thread so the event loop stays responsive
            # during the disk barrier. Without this, a slow fsync
            # (NAS / encrypted volume / SSD GC pause) freezes every
            # coroutine on the loop for the duration — RPC timeouts
            # cannot fire, heartbeat windows expire silently, pool
            # acquirers parked on _pool.get cannot wake.
            #
            # Shield the to_thread await + in-memory assignment
            # together: asyncio cannot abort the worker thread mid-
            # write, so a cancel arriving WHILE the worker is running
            # parks until the future resolves and then re-raises
            # ``CancelledError`` at the await boundary. Pre-fix that
            # cancel could land BETWEEN the worker returning (on-disk
            # state = NEW) and the ``self._nodes = normalised``
            # assignment (in-memory state = OLD), leaving the two
            # divergent for the lifetime of the process. The shield
            # bundles the commit so the cancel only re-raises after
            # both sides have been updated.
            inner = asyncio.ensure_future(self._write_and_publish(normalised, text))
            try:
                await asyncio.shield(inner)
            except asyncio.CancelledError:
                # The outer await was cancelled but the shielded
                # ``inner`` task continues running. Drain it under
                # the lock so the lock remains held until the
                # commit truly finishes — without this, the
                # ``finally`` would release the lock while a
                # concurrent writer could race the orphan.
                # Loop on repeated cancel so a stubborn outer
                # cancel cascade still waits out the worker
                # thread, which asyncio cannot abort.
                while not inner.done():
                    try:
                        await asyncio.shield(inner)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        # Non-cancel inner failure (``OSError`` on a
                        # near-full disk / ``EROFS`` / fsync error
                        # ...). Break so the exception escapes and
                        # supplants the cancel context — the caller
                        # MUST see the disk error rather than a
                        # generic ``CancelledError``. The ``await``
                        # above implicitly observed
                        # ``inner.exception()``; logging at DEBUG
                        # closes the observability gap that previously
                        # left a silent unwind during a cancel cascade
                        # and prevents the asyncio "exception in
                        # shielded future" log from being the only
                        # breadcrumb.
                        logger.debug(
                            "set_nodes: inner write raised during cancel drain",
                            exc_info=True,
                        )
                        raise
                # Defensive: if ``inner`` finished with an exception
                # WITHOUT the drain re-await observing it (e.g. the
                # outer ``except asyncio.CancelledError`` caught a
                # cancel that arrived after inner had already finished
                # with an exception, so the drain loop never entered),
                # read it now to mark the task "retrieved" and prevent
                # the asyncio GC-time "Task exception was never
                # retrieved" warning under ``-X dev`` /
                # ``PYTHONASYNCIODEBUG``. The original
                # ``CancelledError`` is re-raised by ``raise`` below
                # — a non-cancel inner failure observed here is NOT
                # swallowed silently; it would have been caught by the
                # ``except Exception`` arm above if it surfaced
                # during the drain re-await.
                if inner.exception() is not None:
                    logger.debug(
                        "set_nodes: inner write raised before cancel drain entry",
                        exc_info=inner.exception(),
                    )
                raise
        finally:
            self._lock.release()
