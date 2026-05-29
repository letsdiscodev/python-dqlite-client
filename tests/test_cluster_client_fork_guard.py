"""find_leader raises InterfaceError when called after os.fork: the single-flight
slot map holds tasks bound to the parent's loop, so a child would await an
inherited parent-loop task (undefined behaviour)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import InterfaceError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_find_leader_after_fork_raises_interface_error() -> None:
    cluster = ClusterClient(node_store=MemoryNodeStore(["localhost:9001"]))
    fake_parent_pid = cluster._creator_pid + 1
    cluster._creator_pid = fake_parent_pid

    with (
        patch("dqliteclient.connection.os.getpid", return_value=fake_parent_pid + 1),
        pytest.raises(InterfaceError, match="fork"),
    ):
        await cluster.find_leader()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_find_leader_actual_fork_raises() -> None:
    """End-to-end fork: child calls find_leader and reports back via pipe."""
    cluster = ClusterClient(node_store=MemoryNodeStore(["127.0.0.1:9999"]))

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            try:
                import asyncio

                async def run() -> None:
                    await cluster.find_leader()

                asyncio.run(run())
                os.write(w, b"NO_RAISE")
            except InterfaceError as e:
                os.write(w, b"OK" if "fork" in str(e) else f"WRONG_MSG:{e}".encode())
            except Exception as e:  # noqa: BLE001
                os.write(w, f"WRONG_TYPE:{type(e).__name__}:{e}".encode())
            finally:
                os.close(w)
        finally:
            os._exit(0)
    os.close(w)
    result = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        result += chunk
    os.close(r)
    os.waitpid(pid, 0)
    assert result == b"OK", f"child reported: {result!r}"
