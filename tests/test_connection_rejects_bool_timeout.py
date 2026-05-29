"""__init__ must reject bool timeout/close_timeout: bool passes math.isfinite
and ``> 0``, so without an isinstance guard ``timeout=True`` silently means 1s."""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.parametrize("bad", [True, False])
def test_connection_rejects_bool_timeout(bad: bool) -> None:
    with pytest.raises(ValueError, match="bool"):
        DqliteConnection("localhost:9001", timeout=bad)


@pytest.mark.parametrize("bad", [True, False])
def test_connection_rejects_bool_close_timeout(bad: bool) -> None:
    with pytest.raises(ValueError, match="bool"):
        DqliteConnection("localhost:9001", close_timeout=bad)
