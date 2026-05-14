"""Pin: ``YamlNodeStore.set_nodes`` drain loop holds the lock until
the shielded ``inner`` finishes under back-to-back cancels, and
observes a non-cancel inner failure at DEBUG (so the disk-error
event is not silently lost during a cancel cascade).

Background
----------
The cancel-handling path catches ``asyncio.CancelledError`` and then
runs a ``while not inner.done()`` drain loop that re-awaits the
shielded task. Two contracts that were previously uncovered:

1. **Stubborn-recancel coverage**: a second cancel landing while the
   drain re-await is parked must NOT release the lock early. The loop
   re-enters its re-await and the lock stays held until ``inner.done()``.
   The final raise must be ``CancelledError`` (the current re-raise
   behaviour is preserved).

2. **Non-cancel inner exception observed at DEBUG**: if
   ``_write_and_publish`` raises an ``OSError`` (ENOSPC / EROFS /
   fsync failure) during a cancel drain, the exception supplants the
   cancel (the caller still needs to see the disk error) but is also
   logged at DEBUG so the observability gap is closed.
   ``inner.exception()`` is implicitly observed because the drain
   re-await raises it.
"""

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
    """Two back-to-back cancels land: the first surfaces inside
    ``set_nodes`` and enters the drain loop; the second lands while
    the drain re-await is parked. The lock must remain held until
    ``inner`` finishes; only then does the final ``raise`` propagate
    ``CancelledError``."""
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

    # The lock is held while inner is running.
    assert store._lock.locked(), "lock must be held while inner runs"

    # First cancel: lands on the outer ``await asyncio.shield(inner)``.
    task.cancel()
    # Let the cancel propagate up to the drain loop's re-await without
    # releasing the worker yet — the loop must re-await ``inner``.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Second cancel: lands WHILE the drain loop's ``await
    # asyncio.shield(inner)`` is parked. The except arm must continue
    # the loop instead of bubbling out.
    task.cancel()
    await asyncio.sleep(0)

    # Throughout the drain, the lock stays held — releasing it would
    # let a concurrent set_nodes race the still-running worker.
    assert store._lock.locked(), "lock must remain held across stubborn-recancel drain loop"

    # Release the worker — inner completes; drain loop exits; final
    # ``raise`` propagates CancelledError.
    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Sanity: the worker did finish (disk-side commit ran).
    for _ in range(200):
        if finished.is_set():
            break
        await asyncio.sleep(0.01)
    assert finished.is_set(), "worker did not finish slow_replace"

    # Lock must be released after the finally clause.
    assert not store._lock.locked(), "lock must be released after set_nodes returns"


@pytest.mark.asyncio
async def test_inner_oserror_during_drain_observed_at_debug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When ``_write_and_publish`` raises ``OSError`` while a cancel
    is being drained, the drain loop must observe the non-cancel
    failure at DEBUG before letting it propagate. The exception
    still supplants the cancel context (the caller needs to see the
    disk error), but the DEBUG log closes the observability gap that
    previously left a silent unwind.
    """
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
        # Simulate a late rename / fsync failure (ENOSPC, EROFS, etc.).
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

    # Release the worker — inner finishes with OSError; the drain
    # loop re-await re-raises that OSError; the non-cancel except
    # arm observes it at DEBUG and lets it propagate.
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

    # DEBUG log emitted by the drain-loop observation arm.
    debug_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.name == "dqliteclient.node_store"
    ]
    assert any("cancel drain" in m or "drain" in m for m in debug_msgs), (
        f"expected DEBUG log mentioning the drain-loop observation; got {debug_msgs!r}"
    )

    # Lock released even though the cancel + OSError raced.
    assert not store._lock.locked(), "lock must be released after drain"


@pytest.mark.asyncio
async def test_no_cancel_inner_oserror_propagates_to_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary contract: when there is NO cancel and the inner
    write fails with ``OSError``, the caller MUST see the OSError
    (the disk error is not swallowed). This pins the documented
    constraint "do NOT widen the catch to swallow OSError from the
    outer await — caller still needs to see the disk error."
    Failure of this test would mean the fix overreached.
    """
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
    # The original OSError surfaces to the caller — not a CancelledError,
    # not a swallowed-then-None.
    assert exc_info.value.errno == 28

    # Lock released after the finally.
    assert not store._lock.locked(), "lock must be released after OSError propagates"
