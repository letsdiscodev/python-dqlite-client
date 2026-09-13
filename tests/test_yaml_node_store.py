"""YamlNodeStore: loader diagnostics, set_nodes validation, atomic writes, cancel safety."""

from __future__ import annotations

import asyncio
import errno
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole

# Pin: ``_load_from_disk`` caps and sanitises payload-derived values in its
# ``ClusterError`` diagnostics. Without it a corrupt entry can inject CR/LF into logs
# (CWE-117) or place hundreds of KB into ``ClusterError.args[0]`` (uncapped on DqliteError).


def test_role_string_oversize_payload_truncated(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    huge = "A" * 10_000
    yaml_file.write_text(
        "- {ID: 1, Address: 'h:9001', Role: '" + huge + "'}\n",
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" in msg
    assert len(msg) < 2_000


def test_role_string_control_bytes_sanitised(tmp_path: Path) -> None:
    """Control bytes must be escaped so they can't forge fake log lines."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        '- {ID: 1, Address: "h:9001", Role: "foo\\rfaked"}\n',
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "\r" not in msg


def test_address_validate_rewrap_capped(tmp_path: Path) -> None:
    """The ``_validate_and_normalise_nodes`` rewrap also caps+sanitises the inner str(e)."""
    yaml_file = tmp_path / "nodes.yml"
    huge_addr = "x" * 10_000 + ":9000"
    yaml_file.write_text(
        '- {ID: 1, Address: "' + huge_addr + '", Role: voter}\n',
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" in msg
    assert len(msg) < 2_000


def test_happy_path_diagnostic_unchanged(tmp_path: Path) -> None:
    """The cap helper is a no-op for short values."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: 99}\n")

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" not in msg
    assert "99" in msg


# Pin: ``YamlNodeStore`` is picklable (it must not cache the yaml module).


pytest.importorskip("yaml")


def test_yaml_node_store_round_trip_pickle() -> None:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        os.close(fd)
        store = YamlNodeStore(path)
        data = pickle.dumps(store)
        restored = pickle.loads(data)
        assert restored.path == store.path
    finally:
        os.unlink(path)


def test_yaml_node_store_does_not_cache_module_reference() -> None:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        os.close(fd)
        store = YamlNodeStore(path)
        assert not hasattr(store, "_yaml"), (
            "YamlNodeStore should not cache the yaml module on self; "
            "re-import on demand instead so pickle round-trip works."
        )
    finally:
        os.unlink(path)


# Cancel-recovery and tempfile-cleanup invariants for ``YamlNodeStore.set_nodes``.


@pytest.mark.asyncio
async def test_set_nodes_cancel_before_lock_leaves_state_unchanged(
    tmp_path: Path,
) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    original = [NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)]
    await store.set_nodes(original)

    # Park the lock so the cancel delivers on the lock-acquire boundary.
    async with store._lock:
        task = asyncio.create_task(
            store.set_nodes([NodeInfo(node_id=2, address="127.0.0.2:9001", role=NodeRole.VOTER)])
        )
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert tuple(await store.get_nodes()) == tuple(original)
    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    assert tuple(await on_disk.get_nodes()) == tuple(original)
    orphans = list(tmp_path.glob(".nodes.yaml.*.tmp"))
    assert orphans == [], f"orphan tempfile leak after lock-cancel: {orphans}"


@pytest.mark.asyncio
async def test_set_nodes_orphan_tempfile_cleaned_up_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(errno.EINTR, "simulated interrupt")

    monkeypatch.setattr("dqliteclient.node_store.os.replace", failing_replace)

    with pytest.raises(OSError):
        await store.set_nodes([NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)])

    orphans = list(tmp_path.glob(".nodes.yaml.*.tmp"))
    assert orphans == [], (
        f"orphan tempfile leak after os.replace failure: {orphans}; "
        f"the finally: os.unlink cleanup did not run"
    )


@pytest.mark.asyncio
async def test_set_nodes_cancel_during_to_thread_leaves_no_orphan_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    started = threading.Event()
    can_finish = threading.Event()

    real_replace = os.replace
    replace_calls: list[tuple[str, str]] = []

    def slow_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        can_finish.wait(timeout=5.0)
        replace_calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_replace)

    task = asyncio.create_task(
        store.set_nodes([NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)])
    )

    # Poll in the loop (not a synchronous wait) to avoid serialising against
    # the asyncio.to_thread executor.
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker thread did not reach slow_replace within the test budget"

    # Cancel while the worker is parked; asyncio cannot interrupt the thread.
    task.cancel()
    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Filter to paths that still exist: dirent-cache shapes can return entries via
    # ``glob`` that the worker already renamed/unlinked by assert time.
    orphans = [p for p in tmp_path.glob(".nodes.yaml.*.tmp") if p.exists()]
    assert orphans == [], (
        f"orphan tempfile leak after cancel-during-to_thread: {orphans}; "
        f"replace_calls={replace_calls}; "
        f"the finally-arm cleanup or the rename did not converge to "
        f"a self-consistent on-disk state"
    )


# ``set_nodes`` dispatches the atomic-rename ritual to a worker thread so a slow
# fsync does not freeze the event loop.


@pytest.mark.asyncio
async def test_set_nodes_yields_during_slow_fsync(tmp_path: Path) -> None:
    """Block fsync ~500 ms; a concurrent ticker must still tick during set_nodes."""
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)
    new_nodes = [NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)]

    real_fsync = os.fsync

    def slow_fsync(fd: int) -> None:
        time.sleep(0.5)
        real_fsync(fd)

    tick_times: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.02)
            tick_times.append(time.monotonic())

    with patch("dqliteclient.node_store.os.fsync", new=slow_fsync):
        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await store.set_nodes(new_nodes)
        done = time.monotonic()
        stop.set()
        await ticker_task

    ticks_during_write = [t for t in tick_times if started < t < done]
    assert len(ticks_during_write) >= 3, (
        f"only {len(ticks_during_write)} ticks landed during the "
        f"{done - started:.3f}s set_nodes call window; the event loop "
        f"was blocked. Expected the loop to remain responsive via "
        f"asyncio.to_thread dispatch."
    )


@pytest.mark.asyncio
async def test_set_nodes_still_writes_payload_atomically(tmp_path: Path) -> None:
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)
    new_nodes = [
        NodeInfo(node_id=1, address="10.0.0.1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="10.0.0.2:9001", role=NodeRole.SPARE),
    ]
    await store.set_nodes(new_nodes)
    assert store_path.exists()
    text = store_path.read_text(encoding="utf-8")
    assert "10.0.0.1:9001" in text
    assert "10.0.0.2:9001" in text
    assert {n.address for n in await store.get_nodes()} == {
        "10.0.0.1:9001",
        "10.0.0.2:9001",
    }


# ``set_nodes`` runs ``yaml.safe_dump`` in the worker thread, not the loop thread.


@pytest.mark.asyncio
async def test_set_nodes_runs_safe_dump_in_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)

    main_thread_id = threading.get_ident()
    calling_threads: list[int] = []
    real_safe_dump = yaml.safe_dump

    def recording_safe_dump(*args: Any, **kwargs: Any) -> Any:
        calling_threads.append(threading.get_ident())
        return real_safe_dump(*args, **kwargs)

    monkeypatch.setattr(yaml, "safe_dump", recording_safe_dump)

    await store.set_nodes([NodeInfo(node_id=1, address="10.0.0.1:9001", role=NodeRole.VOTER)])

    assert calling_threads, "yaml.safe_dump should have been called"
    on_loop = [tid for tid in calling_threads if tid == main_thread_id]
    assert not on_loop, (
        f"yaml.safe_dump ran on the loop thread (tid={main_thread_id}); "
        f"calling threads recorded: {calling_threads}. The serialisation "
        "must be dispatched via asyncio.to_thread together with the disk "
        "ritual it accompanies."
    )


# ``YamlNodeStore.set_nodes`` enforces the same strip/dedup/``_parse_address``
# pipeline as the loader, so it never writes bytes ``_load_from_disk`` would reject.


def _make_store(path: Path) -> YamlNodeStore:
    return YamlNodeStore(path)


@pytest.mark.asyncio
async def test_set_nodes_rejects_non_string_address() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(TypeError, match="(?i)address must be"):
            await store.set_nodes(
                [NodeInfo(node_id=1, address=12345, role=NodeRole.VOTER)]  # type: ignore[arg-type]
            )


@pytest.mark.asyncio
async def test_set_nodes_rejects_empty_address() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(ValueError, match="(?i)non-empty"):
            await store.set_nodes([NodeInfo(node_id=1, address="", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_rejects_whitespace_only_address() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(ValueError, match="(?i)non-empty"):
            await store.set_nodes([NodeInfo(node_id=1, address="   ", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_rejects_malformed_host_port() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(ValueError, match="host:port"):
            await store.set_nodes(
                [NodeInfo(node_id=1, address="not_a_valid_address", role=NodeRole.VOTER)]
            )


@pytest.mark.asyncio
async def test_set_nodes_strips_whitespace() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        await store.set_nodes(
            [NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER)]
        )
        nodes = await store.get_nodes()
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_set_nodes_dedups_duplicates_first_wins() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        await store.set_nodes(
            [
                NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
                NodeInfo(node_id=2, address="  127.0.0.1:9001  ", role=NodeRole.STANDBY),
                NodeInfo(node_id=3, address="127.0.0.1:9002", role=NodeRole.SPARE),
            ]
        )
        nodes = await store.get_nodes()
        assert len(nodes) == 2
        assert nodes[0].node_id == 1
        assert nodes[1].node_id == 3


@pytest.mark.asyncio
async def test_set_nodes_persists_canonical_form_to_disk() -> None:
    """set_nodes writes canonical form; a fresh store reloads it without rejection."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        store = _make_store(path)
        await store.set_nodes(
            [
                NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER),
                NodeInfo(node_id=2, address="127.0.0.1:9001", role=NodeRole.STANDBY),
            ]
        )
        reloaded = YamlNodeStore(path)
        nodes = await reloaded.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_load_from_disk_strips_and_dedups_via_helper() -> None:
    """``_load_from_disk`` canonicalises hand-edited files via the same pipeline."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        path.write_text(
            "- ID: 1\n"
            "  Address: '  127.0.0.1:9001  '\n"
            "  Role: 0\n"
            "- ID: 2\n"
            "  Address: '127.0.0.1:9001'\n"
            "  Role: 0\n"
        )
        store = YamlNodeStore(path)
        nodes = await store.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_load_from_disk_rejects_invalid_address_via_helper() -> None:
    """A malformed ``host:port`` in a YAML file fails to load via the shared validator."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        path.write_text("- ID: 1\n  Address: 'not_a_valid_address'\n  Role: 0\n")
        from dqliteclient.exceptions import ClusterError

        with pytest.raises(ClusterError, match="host:port"):
            YamlNodeStore(path)


@pytest.mark.asyncio
async def test_concurrent_set_nodes_serialised_under_lock() -> None:
    """Concurrent ``set_nodes`` calls serialise through the lock — no torn writes."""
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")

        async def writer(addr: str) -> None:
            await store.set_nodes([NodeInfo(node_id=1, address=addr, role=NodeRole.VOTER)])

        await asyncio.gather(*(writer(f"127.0.0.{i}:9001") for i in range(1, 11)))
        nodes = await store.get_nodes()
        assert len(nodes) == 1


# ``set_nodes`` shields the to_thread await + in-memory assignment together, so a
# cancel on that boundary cannot leave disk holding NEW membership while ``_nodes``
# holds OLD.


@pytest.mark.asyncio
async def test_cancel_between_to_thread_return_and_assignment_does_not_diverge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Park the worker mid-write, cancel, release; after the cancel surfaces
    in-memory ``_nodes`` must match what landed on disk."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="127.0.0.2:9001", role=NodeRole.VOTER),
    ]

    started = threading.Event()
    can_finish = threading.Event()
    finished = threading.Event()
    real_replace = os.replace

    def slow_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        can_finish.wait(timeout=5.0)
        try:
            real_replace(src, dst)
        finally:
            finished.set()

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_replace)

    task = asyncio.create_task(store.set_nodes(new_nodes))

    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker thread did not enter slow_replace"

    # Deliver the cancel WHILE the worker is parked, then release it.
    task.cancel()
    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Wait for the worker to finish the rename so the disk check sees the commit.
    for _ in range(200):
        if finished.is_set():
            break
        await asyncio.sleep(0.01)
    assert finished.is_set(), "worker thread did not finish slow_replace"

    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    disk_nodes = tuple(await on_disk.get_nodes())
    assert disk_nodes == tuple(new_nodes), (
        f"disk should hold the NEW membership after the rename completed; got {disk_nodes}"
    )

    in_memory = tuple(await store.get_nodes())
    assert in_memory == disk_nodes, (
        f"in-memory _nodes diverged from on-disk after cancel: "
        f"in_memory={in_memory}, on_disk={disk_nodes}. The cancel "
        f"re-raised between the to_thread return and the assignment, "
        f"leaving permanent divergence."
    )
