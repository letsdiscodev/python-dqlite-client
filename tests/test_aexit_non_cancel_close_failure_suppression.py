"""Pin: ``DqliteConnection.__aexit__`` must not let a non-cancel
close-time exception supplant a body exception.

The cancel arm and the non-cancel arm follow the same discipline: a
close-time ``CancelledError`` / ``OSError`` / ``InterfaceError`` must
not supplant an in-flight body exception, because that flips the
*primary* exception class as seen by ``except``-by-class consumers.

This test pins the symmetric fix: the body's ``ValueError("body")``
propagates to the caller, and the close-time ``OSError("close")`` is
captured at DEBUG level on the ``dqliteclient.connection`` logger.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_aexit_close_oserror_does_not_supplant_body_valueerror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Body raises ``ValueError("body")``; ``close()`` raises
    ``OSError("close")``. Caller must observe ``ValueError("body")``,
    and the close-time error must be DEBUG-logged."""
    conn = DqliteConnection("localhost:9001", database="test", timeout=5.0)

    # Patch close() to simulate a close-time non-cancel failure.
    with (
        patch.object(
            DqliteConnection,
            "close",
            new=AsyncMock(side_effect=OSError("close")),
        ),
        caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"),
        pytest.raises(ValueError, match="body"),
    ):
        async with conn:
            raise ValueError("body")

    # The close-time OSError should be DEBUG-logged with a useful marker.
    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and "close" in r.getMessage().lower()
        and "__aexit__" in r.getMessage()
    ]
    assert matching, (
        "expected a DEBUG log on dqliteclient.connection mentioning the "
        "close-time error during __aexit__; got "
        f"{[(r.levelno, r.getMessage()) for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_aexit_clean_body_close_oserror_propagates() -> None:
    """Regression guard: when the body exits cleanly, a close-time
    ``OSError`` must still propagate (no body exception to preserve)."""
    conn = DqliteConnection("localhost:9001", database="test", timeout=5.0)

    with (
        patch.object(
            DqliteConnection,
            "close",
            new=AsyncMock(side_effect=OSError("close")),
        ),
        pytest.raises(OSError, match="close"),
    ):
        async with conn:
            pass
