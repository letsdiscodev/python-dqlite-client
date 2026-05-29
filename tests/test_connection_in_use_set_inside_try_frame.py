"""``connect`` and ``close`` set ``self._in_use = True`` INSIDE the try-frame
(like ``_run_protocol``): a KeyboardInterrupt / SystemExit on the bytecode boundary
before the ``try:`` would leave ``_in_use`` stuck True for the process lifetime."""

from __future__ import annotations

import inspect

from dqliteclient import connection as connection_module


def _source_of(name: str) -> str:
    obj = getattr(connection_module.DqliteConnection, name)
    return inspect.getsource(obj)


def _try_open_index(src: str) -> int:
    """Index of the FIRST ``try:`` in ``src``."""
    needle = "try:"
    return src.find(needle)


def _in_use_assignment_index(src: str) -> int:
    needle = "self._in_use = True"
    return src.find(needle)


def test_connect_in_use_assignment_inside_try() -> None:
    """``connect()`` must claim ``_in_use`` AFTER the ``try:`` so an escaping
    exception lands in the except/finally arm that clears the flag."""
    src = _source_of("connect")
    try_pos = _try_open_index(src)
    assign_pos = _in_use_assignment_index(src)
    assert try_pos != -1, "connect() must contain a try-block"
    assert assign_pos != -1, "connect() must contain self._in_use = True"
    assert try_pos < assign_pos, (
        "connect() must set self._in_use = True INSIDE the try-frame so a "
        "KeyboardInterrupt / SystemExit cannot escape with _in_use stuck True."
    )


def test_close_in_use_assignment_inside_try() -> None:
    """``close()`` must follow the same discipline as ``connect()``."""
    src = _source_of("close")
    try_pos = _try_open_index(src)
    assign_pos = _in_use_assignment_index(src)
    assert try_pos != -1, "close() must contain a try-block"
    assert assign_pos != -1, "close() must contain self._in_use = True"
    assert try_pos < assign_pos, (
        "close() must set self._in_use = True INSIDE the try-frame; "
        "see connect() and _run_protocol siblings."
    )


def test_run_protocol_pin_unchanged() -> None:
    """Negative control: ``_run_protocol`` (the shape we mirror) must keep the
    flag inside the try."""
    src = _source_of("_run_protocol")
    try_pos = _try_open_index(src)
    assign_pos = _in_use_assignment_index(src)
    assert try_pos != -1
    assert assign_pos != -1
    assert try_pos < assign_pos
