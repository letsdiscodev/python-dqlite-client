"""``DqliteConnection.__init__`` must reject ``bool`` for ``timeout`` /
``close_timeout``.

``bool`` subclasses ``int`` and ``isinstance(True, float)`` is False,
but ``math.isfinite(True)`` returns True and ``True > 0`` is True —
without an explicit isinstance guard, ``timeout=True`` silently gives
a 1-second budget. Matches the sibling
``_validate_positive_int_or_none`` in ``protocol.py``.
"""

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
