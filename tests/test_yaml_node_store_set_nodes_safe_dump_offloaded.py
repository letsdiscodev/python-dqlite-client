"""Pin: ``YamlNodeStore.set_nodes`` runs ``yaml.safe_dump`` inside the
``asyncio.to_thread`` worker, not on the event loop thread.

The function's own comment block already implies the serialisation is
offloaded: ``_write_atomic_sync``'s docstring states "``yaml.safe_dump``
is cheap CPU, but ``os.fsync`` against a contended disk ... can stall
for tens to hundreds of milliseconds" — describing the offloaded ritual
as if ``safe_dump`` were part of it. Pre-fix the call sat on the loop
thread and only the disk I/O was dispatched, so a large-cluster payload
(e.g. 1000 nodes ~ 10 ms of pure Python) would stall the loop before
the worker hop. Move ``yaml.safe_dump`` into the worker so the
documented contract holds.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_runs_safe_dump_in_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``yaml.safe_dump`` must run on the worker thread dispatched by
    ``asyncio.to_thread``, not on the loop thread.
    """
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)

    main_thread_id = threading.get_ident()
    calling_threads: list[int] = []
    real_safe_dump = yaml.safe_dump

    def recording_safe_dump(*args: Any, **kwargs: Any) -> Any:
        calling_threads.append(threading.get_ident())
        return real_safe_dump(*args, **kwargs)

    # Monkeypatch the module-level ``yaml.safe_dump`` so every call
    # site (the store's worker-thread dispatch is the only one
    # exercised here) records its calling thread.
    monkeypatch.setattr(yaml, "safe_dump", recording_safe_dump)

    await store.set_nodes([NodeInfo(node_id=1, address="10.0.0.1:9001", role=NodeRole.VOTER)])

    assert calling_threads, "yaml.safe_dump should have been called"
    # Every recorded call must be on a thread distinct from the loop
    # thread. Even a single call on the main thread means the
    # serialisation cost was paid on the loop.
    on_loop = [tid for tid in calling_threads if tid == main_thread_id]
    assert not on_loop, (
        f"yaml.safe_dump ran on the loop thread (tid={main_thread_id}); "
        f"calling threads recorded: {calling_threads}. The serialisation "
        "must be dispatched via asyncio.to_thread together with the disk "
        "ritual it accompanies."
    )
