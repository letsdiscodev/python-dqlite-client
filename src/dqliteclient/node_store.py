"""Node store interfaces for cluster discovery."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from dqliteclient.exceptions import ClusterError
from dqlitewire import NodeRole

__all__ = ["MemoryNodeStore", "NodeInfo", "NodeStore", "YamlNodeStore"]


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
    ``dqliteclient.connection._parse_address`` — hostnames are
    ASCII-only and lowercased, IP literals are canonicalised."""

    role: NodeRole
    """Raft role: ``VOTER``, ``STANDBY``, or ``SPARE``. Standby and
    spare nodes cannot become leader —
    ``ClusterClient.find_leader`` sorts voters first before probing."""


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
    """

    def __init__(
        self,
        addresses: list[str] | None = None,
        *,
        initial_addresses: list[str] | None = None,
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
        seed = addresses if addresses is not None else initial_addresses
        if seed:
            # Strip leading/trailing whitespace and reject empty
            # entries up front. Without this, a typoed seed like
            # ``["localhost:9001\n"]`` (e.g. read from a config
            # file) leaks ``ValueError`` through the sweep's narrow
            # exception filter and surfaces deep inside
            # ``find_leader``. Deduplicate while preserving order
            # so a config with the same address twice doesn't
            # double the probe count and double the per-node
            # error lines in the failure-aggregate message.
            seen: set[str] = set()
            unique: list[str] = []
            for raw in seed:
                if not isinstance(raw, str):
                    raise TypeError(
                        f"NodeStore addresses must be 'host:port' strings, got {type(raw).__name__}"
                    )
                addr = raw.strip()
                if not addr:
                    raise ValueError("NodeStore addresses must be non-empty 'host:port' strings")
                # Syntactic validation via the same parser
                # ``_query_leader`` will use later. Without this,
                # malformed entries (unbracketed IPv6, non-numeric
                # port, IDN, credentials shape) leak ``ValueError``
                # through the ``find_leader`` sweep's narrow except
                # tuple — aborting the entire sweep with the wrong
                # exception class and no per-node attribution.
                # Local import to avoid cluster <-> node_store
                # circular import (cluster imports NodeStore).
                from dqliteclient.connection import _parse_address as _parse_addr_validator

                try:
                    _parse_addr_validator(addr)
                except ValueError as e:
                    raise ValueError(
                        f"NodeStore address {addr!r} is not a valid 'host:port': {e}"
                    ) from e
                if addr in seen:
                    continue
                seen.add(addr)
                unique.append(addr)
            self._nodes: tuple[NodeInfo, ...] = tuple(
                NodeInfo(node_id=i + 1, address=addr, role=NodeRole.VOTER)
                for i, addr in enumerate(unique)
            )
        else:
            self._nodes = ()

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Get list of known nodes."""
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes.

        Mirrors the strip / dedup / empty-rejection validation done
        in ``__init__``. Without this, a runtime update with a
        whitespace-laden or duplicated address would leak through
        and surface deep inside ``find_leader`` (whitespace) or
        inflate the probe count and per-node error lines
        (duplicates).
        """
        seen: set[str] = set()
        unique: list[NodeInfo] = []
        for node in nodes:
            if not isinstance(node.address, str):
                raise TypeError(
                    f"NodeInfo.address must be 'host:port' string, got "
                    f"{type(node.address).__name__}"
                )
            addr = node.address.strip()
            if not addr:
                raise ValueError("NodeInfo.address must be a non-empty 'host:port' string")
            # Same syntactic validation as ``__init__``; see rationale
            # there. Local import to avoid the cluster import cycle.
            from dqliteclient.connection import _parse_address as _parse_addr_validator

            try:
                _parse_addr_validator(addr)
            except ValueError as e:
                raise ValueError(
                    f"NodeInfo.address {addr!r} is not a valid 'host:port': {e}"
                ) from e
            if addr in seen:
                continue
            seen.add(addr)
            if addr is node.address:
                unique.append(node)
            else:
                # NodeInfo is a frozen dataclass; rebuild with the
                # canonical (stripped) address so a downstream lookup
                # by-address matches the canonical form.
                unique.append(NodeInfo(node_id=node.node_id, address=addr, role=node.role))
        self._nodes = tuple(unique)


# Role-string aliases accepted on read for ergonomics with hand-edited
# files. ``write()`` always emits the canonical integer Role enum
# value, matching go-dqlite's ``yaml.v2`` output of ``NodeRole int``.
# Normalisation strips dashes and underscores so ``stand-by``,
# ``stand_by``, and ``standby`` all map to the same role.
_YAML_ROLE_STRING_ALIASES: dict[str, NodeRole] = {
    "voter": NodeRole.VOTER,
    "standby": NodeRole.STANDBY,
    "spare": NodeRole.SPARE,
}


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

    Format pinning: this implementation tracks go-dqlite ``d046c95``.
    Any future upstream format change must be reflected here AND in
    the interop test fixture.

    PyYAML is an optional dependency: install
    ``python-dqlite-client[yaml-store]`` to use this class. The base
    install does not pull PyYAML.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        # Defer-import so PyYAML stays an optional extra. Operators
        # who don't use YamlNodeStore never need the dependency.
        try:
            import yaml as _yaml
        except ImportError as e:
            raise ImportError(
                "YamlNodeStore requires PyYAML; install with "
                "'pip install python-dqlite-client[yaml-store]'"
            ) from e
        self._yaml = _yaml
        self._path = Path(path)
        self._lock = asyncio.Lock()
        # Eager-load and validate so a malformed file raises at
        # construction (operator-facing) rather than at first
        # ``get_nodes`` (deep inside ``find_leader``).
        self._nodes: tuple[NodeInfo, ...] = self._load_from_disk()

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
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as e:
            raise ClusterError(f"YamlNodeStore: cannot read {self._path}: {e}") from e
        if not text.strip():
            return ()
        try:
            raw = self._yaml.safe_load(text)
        except self._yaml.YAMLError as e:
            raise ClusterError(f"YamlNodeStore: malformed YAML in {self._path}: {e}") from e
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ClusterError(
                f"YamlNodeStore: {self._path} top-level must be a "
                f"YAML list, got {type(raw).__name__}"
            )
        # Local import to avoid cluster <-> node_store circular import
        # (cluster imports NodeStore).
        from dqliteclient.connection import _parse_address as _parse_addr_validator

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
            try:
                _parse_addr_validator(address)
            except ValueError as e:
                raise ClusterError(
                    f"YamlNodeStore: {self._path}[{idx}] address {address!r} invalid: {e}"
                ) from e
            result.append(NodeInfo(node_id=node_id, address=address, role=role))
        return tuple(result)

    async def get_nodes(self) -> Sequence[NodeInfo]:
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        # Serialise concurrent writes — file rename is atomic but two
        # writers racing the load -> serialise -> write sequence
        # would lose one update. asyncio.Lock keeps the in-process
        # last-writer-wins ordering deterministic.
        async with self._lock:
            payload = [
                {
                    "ID": int(n.node_id),
                    "Address": str(n.address),
                    "Role": int(n.role),  # integer enum value (go-dqlite shape)
                }
                for n in nodes
            ]
            text = self._yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
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
                # Match go-dqlite's 0600 mode before the rename so the
                # final file's mode is correct atomically.
                os.chmod(fd_path, 0o600)
                os.replace(fd_path, self._path)
                fd_path = None  # ownership transferred
                # Sync the parent directory so the rename's metadata
                # block is durable across a hard reboot. POSIX rename
                # is atomic for visibility but the directory entry
                # change can revert after a power loss / kernel
                # panic / VM hard-stop without the directory fsync.
                # Mirrors go-dqlite's ``renameio.WriteFile`` which
                # does this. Best-effort: non-POSIX (Windows) and
                # some FUSE filesystems don't support fsync on
                # directories — suppress OSError so the rename is
                # still visible even if the durability barrier
                # isn't available.
                with contextlib.suppress(OSError):
                    dir_fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                self._nodes = tuple(nodes)
            finally:
                if fd_path is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(fd_path)
