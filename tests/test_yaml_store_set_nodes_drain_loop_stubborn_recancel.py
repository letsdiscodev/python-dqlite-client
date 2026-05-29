"""``set_nodes``'s cancel drain loop holds the lock until the shielded ``inner``
finishes under back-to-back cancels, and observes a non-cancel inner failure at DEBUG
so a disk error is not silently lost during a cancel cascade."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_stubborn_recancel_keeps_lock_until_inner_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second cancel landing during the drain re-await must not release the lock
    early; the lock stays held until ``inner`` finishes, then ``CancelledError`` raises."""
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

    # Wait for the worker thread to enter the parked replace.
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker did not enter slow_replace"

    assert store._lock.locked(), "lock must be held while inner runs"

    # First cancel lands on the outer ``await asyncio.shield(inner)``; let it
    # propagate to the drain re-await without releasing the worker yet.
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Second cancel lands while the drain re-await is parked; the except arm
    # must continue the loop instead of bubbling out.
    task.cancel()
    await asyncio.sleep(0)

    assert store._lock.locked(), "lock must remain held across stubborn-recancel drain loop"

    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(200):
        if finished.is_set():
            break
        await asyncio.sleep(0.01)
    assert finished.is_set(), "worker did not finish slow_replace"

    assert not store._lock.locked(), "lock must be released after set_nodes returns"


@pytest.mark.asyncio
async def test_inner_oserror_during_drain_observed_at_debug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ``OSError`` raised during a cancel drain is logged at DEBUG and then
    propagates, supplanting the cancel so the caller still sees the disk error."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]

    started = threading.Event()
    can_finish = threading.Event()

    class _Boom(OSError):
        pass

    def slow_then_raising_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        can_finish.wait(timeout=5.0)
        raise _Boom(28, "simulated disk-full at rename")

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_then_raising_replace)

    task = asyncio.create_task(store.set_nodes(new_nodes))

    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker did not enter slow_then_raising_replace"

    # Cancel the outer await — drain loop entered.
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    can_finish.set()

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.node_store"),
        pytest.raises(OSError) as exc_info,
    ):
        await task

    assert exc_info.value.errno == 28, (
        "non-cancel inner exception must supplant the cancel context "
        "so the caller sees the disk error"
    )

    debug_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.name == "dqliteclient.node_store"
    ]
    assert any("cancel drain" in m or "drain" in m for m in debug_msgs), (
        f"expected DEBUG log mentioning the drain-loop observation; got {debug_msgs!r}"
    )

    assert not store._lock.locked(), "lock must be released after drain"


@pytest.mark.asyncio
async def test_set_nodes_observes_inner_exception_set_before_outer_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When ``inner`` is done with an exception before the cancel is caught, the drain
    ``while`` never enters; the defensive arm reads ``inner.exception()`` (marking it
    retrieved, avoiding 'Task exception was never retrieved') and logs at DEBUG."""
    import asyncio as _asyncio_top

    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]

    class _Boom(OSError):
        pass

    async def raising_write_and_publish(*_args: object, **_kwargs: object) -> None:
        raise _Boom(28, "simulated disk-full before drain entry")

    monkeypatch.setattr(YamlNodeStore, "_write_and_publish", raising_write_and_publish)

    real_shield = _asyncio_top.shield
    shield_calls: list[int] = []

    async def fake_shield(arg: object, *, loop: object = None) -> None:
        # First call: drive inner to completion with its exception, then raise
        # CancelledError so inner.done() is True when the outer cancel arm runs
        # and the drain while-body never enters.
        shield_calls.append(1)
        if len(shield_calls) == 1:
            assert isinstance(arg, _asyncio_top.Future)
            done, _ = await _asyncio_top.wait({arg})
            assert arg.done()
            assert arg.exception() is not None
            raise _asyncio_top.CancelledError()
        return await real_shield(arg)  # type: ignore[arg-type]

    monkeypatch.setattr("dqliteclient.node_store.asyncio.shield", fake_shield)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.node_store")

    # set_nodes re-raises CancelledError, not the inner OSError.
    with pytest.raises(_asyncio_top.CancelledError):
        await store.set_nodes(new_nodes)

    debug_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.name == "dqliteclient.node_store"
    ]
    assert any("before cancel drain entry" in m for m in debug_msgs), (
        f"expected DEBUG log mentioning 'before cancel drain entry' from the "
        f"defensive observe arm; got {debug_msgs!r}"
    )

    assert not store._lock.locked(), "lock must be released after defensive observe arm fires"


@pytest.mark.asyncio
async def test_no_cancel_inner_oserror_propagates_to_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no cancel, an inner ``OSError`` must reach the caller (not be swallowed)."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]

    class _Boom(OSError):
        pass

    def raising_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        raise _Boom(28, "simulated disk-full at rename")

    monkeypatch.setattr("dqliteclient.node_store.os.replace", raising_replace)

    with pytest.raises(OSError) as exc_info:
        await store.set_nodes(new_nodes)
    assert exc_info.value.errno == 28

    assert not store._lock.locked(), "lock must be released after OSError propagates"
