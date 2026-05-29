"""``set_nodes`` runs ``yaml.safe_dump`` in the worker thread, not the loop thread."""

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
