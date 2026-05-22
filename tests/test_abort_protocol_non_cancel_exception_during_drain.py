"""Pin: ``DqliteConnection._abort_protocol``'s ``except Exception``
arm at connection.py:2084-2089 absorbs non-OSError, non-cancel
exceptions raised by the shielded drain and DEBUG-logs them.

The arm exists for transport-noise scenarios (a half-torn writer's
``wait_closed`` surfacing ``RuntimeError("Transport is closed")``
or similar). Existing tests cover the OSError arm
(``test_abort_protocol_address_log.py``) and the cancel-shield
happy path (``test_abort_protocol_cancel_shield.py``); the
non-cancel ``except Exception`` arm post-shield was previously
uncovered.

Sharpening per the reviewer's note: the stub must raise the
``RuntimeError`` from the INNER drain (not the outer await) so the
shield-wrapped Task captures it. The done-callback then observes
the stored exception, the ``except Exception`` arm DEBUG-logs, and
no ``"Task exception was never retrieved"`` warning lands at GC.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


async def test_abort_protocol_non_cancel_exception_from_inner_drain_swallowed_and_debug_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The inner drain (``await wait_closed``) raises a non-cancel
    ``RuntimeError``; the ``except Exception`` arm absorbs it and
    DEBUG-logs. No warning at GC.
    """
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._close_timeout = 5.0
    conn._address = "localhost:9001"
    proto = MagicMock()
    proto.close = MagicMock()

    async def _raises_runtime_error() -> None:
        # Yield once so the shield wraps the resulting Task before the
        # raise is observed — exercise the post-shield path.
        await asyncio.sleep(0)
        raise RuntimeError("Transport is closed")

    proto.wait_closed = AsyncMock(side_effect=_raises_runtime_error)
    conn._protocol = proto

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        # Must NOT raise — the Exception arm absorbs the inner raise.
        await conn._abort_protocol()
        for _ in range(10):
            await asyncio.sleep(0)
        gc.collect()

    debug_lines = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelname == "DEBUG"
        and "_abort_protocol: unexpected drain error" in rec.getMessage()
    ]
    assert debug_lines, (
        "the except Exception arm must DEBUG-log the inner drain "
        f"exception; got log records {[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
    )
    pending_task_warnings = [
        str(w.message)
        for w in caught
        if "Task exception was never retrieved" in str(w.message)
        or ("destroyed" in str(w.message).lower() and "pending" in str(w.message).lower())
    ]
    assert pending_task_warnings == [], (
        f"shield's done-callback discipline must observe the inner "
        f"exception; got orphan warnings {pending_task_warnings!r}"
    )
    # Sync close ran.
    proto.close.assert_called_once_with()
