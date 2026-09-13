"""Node stores: where cluster discovery gets its seed addresses."""

import asyncio
import contextlib
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, final, runtime_checkable

from dqliteclient._validate import get_current_pid, parse_address
from dqliteclient.exceptions import ClusterError, InterfaceError
from dqlitewire import MAX_NODE_COUNT, NodeRole

__all__ = ["MemoryNodeStore", "NodeInfo", "NodeStore", "YamlNodeStore"]

_ROLE_ALIASES = {"voter": NodeRole.VOTER, "standby": NodeRole.STANDBY, "spare": NodeRole.SPARE}
_MAX_FILE_BYTES = 1 << 20


@final
@dataclass(frozen=True, slots=True)
class NodeInfo:
    """A cluster member: raft node id (1-based), ``host:port`` address, role."""

    node_id: int
    address: str
    role: NodeRole

    def __post_init__(self) -> None:
        if not isinstance(self.role, NodeRole):
            try:
                object.__setattr__(self, "role", NodeRole(self.role))
            except ValueError as e:
                raise ValueError(
                    f"NodeInfo: unknown role {self.role!r}; valid roles are 0 (VOTER), "
                    "1 (STANDBY), 2 (SPARE)"
                ) from e
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, int):
            raise TypeError(f"NodeInfo.node_id must be int, got {type(self.node_id).__name__}")
        if self.node_id < 1:
            raise ValueError(f"NodeInfo.node_id must be >= 1, got {self.node_id}")


@runtime_checkable
class NodeStore(Protocol):
    """Where the cluster client reads and records the known nodes."""

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Return a snapshot the caller must not mutate."""
        ...

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None: ...


def normalise_nodes(nodes: Iterable[NodeInfo]) -> tuple[NodeInfo, ...]:
    """Strip addresses, validate them, and drop duplicates by canonical ``(host, port)``."""
    seen: set[tuple[str, int]] = set()
    unique: list[NodeInfo] = []
    for node in nodes:
        if not isinstance(node.address, str):
            raise TypeError(f"NodeInfo.address must be a str, got {type(node.address).__name__}")
        address = node.address.strip()
        if not address:
            raise ValueError("NodeInfo.address must be a non-empty 'host:port' string")
        try:
            canonical = parse_address(address)
        except ValueError as e:
            raise ValueError(f"NodeInfo.address {address!r} is not a valid 'host:port': {e}") from e
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(
            node if address == node.address else NodeInfo(node.node_id, address, node.role)
        )
        if len(unique) > MAX_NODE_COUNT:
            raise ValueError(f"too many nodes: more than {MAX_NODE_COUNT}")
    return tuple(unique)


class _ForkGuard:
    def __init__(self) -> None:
        self._pid = os.getpid()

    def _check_pid(self) -> None:
        if get_current_pid() != self._pid:
            raise InterfaceError(
                f"{type(self).__name__} used after fork; reconstruct it in the child process "
                f"(created in pid {self._pid}, current pid {get_current_pid()})"
            )


def _snippet(value: object, limit: int = 200) -> str:
    """``repr`` of a payload-derived value, control characters escaped, length capped."""
    text = repr(value)
    if len(text) > limit:
        text = f"{text[: limit // 2]}...{text[-limit // 2 :]} (truncated)"
    return text.encode("unicode_escape").decode("ascii")


class MemoryNodeStore(_ForkGuard):
    """In-memory store seeded from ``host:port`` strings (all recorded as voters)."""

    def __init__(self, addresses: Sequence[str] | None = None) -> None:
        super().__init__()
        seeds: list[NodeInfo] = []
        for raw in addresses or ():
            if not isinstance(raw, str):
                raise TypeError(f"addresses must be 'host:port' strings, got {type(raw).__name__}")
            seeds.append(NodeInfo(len(seeds) + 1, raw, NodeRole.VOTER))
        self._nodes = normalise_nodes(seeds)

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"cannot pickle {type(self).__name__!r}; rebuild it from its seed instead")

    async def get_nodes(self) -> Sequence[NodeInfo]:
        self._check_pid()
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        self._check_pid()
        self._nodes = normalise_nodes(nodes)


class YamlNodeStore(_ForkGuard):
    """File-backed store in go-dqlite's YAML shape::

        - ID: 1
          Address: node1:9001
          Role: 0

    Lower-case keys and role names (``voter`` / ``standby`` / ``spare``) are accepted on
    read; writes are atomic (temp file + rename, mode 0600). Requires PyYAML.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__()
        _require_yaml()
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._nodes = self._load()

    def __reduce__(self) -> tuple[type["YamlNodeStore"], tuple[str]]:
        return (type(self), (str(self._path),))

    @classmethod
    async def create(cls, path: str | os.PathLike[str]) -> "YamlNodeStore":
        """Construct without blocking the loop on the initial file read."""
        return await asyncio.to_thread(cls, path)

    @property
    def path(self) -> Path:
        return self._path

    async def get_nodes(self) -> Sequence[NodeInfo]:
        self._check_pid()
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        self._check_pid()
        normalised = normalise_nodes(nodes)
        payload = [{"ID": n.node_id, "Address": n.address, "Role": int(n.role)} for n in normalised]
        async with self._lock:
            write = asyncio.ensure_future(asyncio.to_thread(self._write, payload))

            def publish(done: asyncio.Future[None]) -> None:
                if not done.cancelled() and done.exception() is None:
                    self._nodes = normalised

            # The thread cannot be interrupted, so the in-memory view follows the file
            # even if the awaiting task is cancelled mid-write.
            write.add_done_callback(publish)
            await asyncio.shield(write)

    def _load(self) -> tuple[NodeInfo, ...]:
        try:
            if self._path.stat().st_size > _MAX_FILE_BYTES:
                raise ClusterError(f"YamlNodeStore: {self._path} exceeds {_MAX_FILE_BYTES} bytes")
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ()
        except (OSError, UnicodeDecodeError) as e:
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
            raise ClusterError(f"YamlNodeStore: {self._path}: top-level must be a YAML list")
        try:
            return normalise_nodes(_parse_entry(self._path, i, e) for i, e in enumerate(raw))
        except (TypeError, ValueError) as e:
            raise ClusterError(f"YamlNodeStore: {self._path}: {_snippet(str(e))}") from e

    def _write(self, payload: list[dict[str, Any]]) -> None:
        import yaml

        text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            delete=False,
        ) as tmp:
            try:
                tmp.write(text)
                tmp.flush()
                os.fsync(tmp.fileno())
            except BaseException:
                os.unlink(tmp.name)
                raise
        try:
            os.replace(tmp.name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)
            raise


def _require_yaml() -> None:
    try:
        import yaml  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "YamlNodeStore requires PyYAML; install python-dqlite-client[yaml-store]"
        ) from e


def _field(entry: dict[str, Any], key: str) -> Any:
    """go-dqlite writes PascalCase keys; hand-edited files may use lower-case ones."""
    value = entry.get(key)
    return value if value is not None else entry.get(key.lower())


def _parse_entry(path: Path, index: int, entry: object) -> NodeInfo:
    where = f"YamlNodeStore: {path}[{index}]"
    if not isinstance(entry, dict):
        raise ClusterError(f"{where} must be a mapping, got {type(entry).__name__}")
    node_id = _field(entry, "ID")
    address = _field(entry, "Address")
    role_raw = _field(entry, "Role")
    if node_id is None:
        raise ClusterError(f"{where} missing 'ID'")
    if address is None:
        raise ClusterError(f"{where} missing 'Address'")
    if isinstance(node_id, str) and node_id.strip().isdigit():
        node_id = int(node_id)
    if isinstance(node_id, bool) or not isinstance(node_id, int):
        raise ClusterError(f"{where} 'ID' must be integer, got {_snippet(node_id)}")
    if not isinstance(address, str):
        raise ClusterError(f"{where} 'Address' must be str, got {type(address).__name__}")
    if role_raw is None:
        role: NodeRole = NodeRole.VOTER
    elif isinstance(role_raw, int) and not isinstance(role_raw, bool):
        try:
            role = NodeRole(role_raw)
        except ValueError as e:
            raise ClusterError(f"{where} 'Role' {role_raw} is not a valid NodeRole") from e
    elif isinstance(role_raw, str):
        key = role_raw.strip().lower().replace("-", "").replace("_", "")
        if key not in _ROLE_ALIASES:
            raise ClusterError(
                f"{where} 'Role' {_snippet(role_raw)} is not one of voter/stand-by/spare"
            )
        role = _ROLE_ALIASES[key]
    else:
        raise ClusterError(f"{where} 'Role' must be int or str, got {type(role_raw).__name__}")
    try:
        return NodeInfo(node_id, address, role)
    except (TypeError, ValueError) as e:
        raise ClusterError(f"{where}: {_snippet(str(e))}") from e
