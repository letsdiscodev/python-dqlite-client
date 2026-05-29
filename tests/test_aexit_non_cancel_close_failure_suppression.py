"""__aexit__ must not let a non-cancel close-time exception supplant a body exception.

Supplanting would flip the primary exception class seen by except-by-class consumers.
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
    """Body ValueError propagates; close-time OSError is DEBUG-logged, not raised."""
    conn = DqliteConnection("localhost:9001", database="test", timeout=5.0)

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
    """When the body exits cleanly, a close-time OSError must still propagate."""
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
