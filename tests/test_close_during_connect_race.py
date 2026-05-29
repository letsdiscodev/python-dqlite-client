"""close() during an in-flight connect() must raise InterfaceError, not
silently early-return on ``_protocol is None`` and leak the connect handle."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


class TestCloseDuringConnect:
    async def test_close_during_inflight_connect_raises_in_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task A is suspended inside connect(); Task B's close() must raise."""
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()

        async def _wait_closed() -> None:
            return None

        writer.wait_closed = MagicMock(side_effect=_wait_closed)

        open_in_flight = asyncio.Event()
        may_finish = asyncio.Event()

        async def _fake_open_connection(
            host: str, port: int, **_kwargs: object
        ) -> tuple[object, object]:
            open_in_flight.set()
            await may_finish.wait()
            return reader, writer

        monkeypatch.setattr("asyncio.open_connection", _fake_open_connection)

        conn = DqliteConnection("localhost:19001", timeout=5.0)

        async def task_a() -> None:
            with contextlib.suppress(Exception):
                await conn.connect()

        connect_task = asyncio.create_task(task_a())
        await open_in_flight.wait()
        # Let connect() set _in_use=True and suspend at the open_connection await.
        await asyncio.sleep(0)
        assert conn._in_use is True
        assert conn._protocol is None

        with pytest.raises(InterfaceError, match="in progress|in use"):
            await conn.close()

        # Let connect finish so the task is not orphaned.
        may_finish.set()
        connect_task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await connect_task
