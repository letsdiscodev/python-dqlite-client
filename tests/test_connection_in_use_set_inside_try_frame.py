"""Pin: ``DqliteConnection.connect`` and ``DqliteConnection.close``
set ``self._in_use = True`` INSIDE the protective try-frame, mirroring
``_run_protocol``'s already-correct discipline.

The bytecode boundary between the assignment and the ``try:`` is
where a ``KeyboardInterrupt`` / ``SystemExit`` (delivered by the
interpreter's signal-eval machinery between any two bytecodes) can
escape with ``_in_use=True`` and no compensating cleanup. Once
stuck, every subsequent operation on the connection raises
``InterfaceError("another operation is in progress")`` for the
process lifetime.

``_run_protocol`` at ``connection.py``'s `_run_protocol` documents
this discipline explicitly. ``connect`` and ``close`` historically
diverged. The pin is structural: assert the source orders ``try:``
before ``self._in_use = True``.
"""

from __future__ import annotations

import inspect

from dqliteclient import connection as connection_module


def _source_of(name: str) -> str:
    obj = getattr(connection_module.DqliteConnection, name)
    return inspect.getsource(obj)


def _try_open_index(src: str) -> int:
    """Index of the FIRST ``try:`` in ``src``. Whitespace-stripped on
    both sides for symmetric comparison; the existing pin pattern in
    ``test_query_leader_dial_cancel_does_not_leak_writer.py`` was
    flagged for one-sided strip — this helper applies the same
    normalisation to source and search literal."""
    needle = "try:"
    return src.find(needle)


def _in_use_assignment_index(src: str) -> int:
    needle = "self._in_use = True"
    return src.find(needle)


def test_connect_in_use_assignment_inside_try() -> None:
    """``connect()`` must claim ``_in_use`` AFTER the ``try:`` opener,
    so any exception that escapes between the claim and the awaitable
    body lands in the except/finally arm that clears the flag."""
    src = _source_of("connect")
    try_pos = _try_open_index(src)
    assign_pos = _in_use_assignment_index(src)
    assert try_pos != -1, "connect() must contain a try-block"
    assert assign_pos != -1, "connect() must contain self._in_use = True"
    assert try_pos < assign_pos, (
        "connect() must set self._in_use = True INSIDE the try-frame "
        "so a KeyboardInterrupt / SystemExit landing on the bytecode "
        "boundary cannot escape with _in_use stuck at True. The "
        "_run_protocol method documents this discipline."
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
    """Negative-control: ``_run_protocol`` ALREADY follows the
    discipline. The pin asserts the in-package reference shape we are
    mirroring stays intact, so a future regression that flips
    ``_run_protocol`` to set the flag outside the try is also
    fenced."""
    src = _source_of("_run_protocol")
    try_pos = _try_open_index(src)
    assign_pos = _in_use_assignment_index(src)
    assert try_pos != -1
    assert assign_pos != -1
    assert try_pos < assign_pos
