"""Pin: the client-layer ``_is_no_tx_rollback_error`` recogniser
shares its substring list with ``dqlitewire.NO_TRANSACTION_MESSAGE_SUBSTRINGS``.

A previous shape inlined the literal substrings in two places
(client and dbapi). Centralising the source in the wire layer
makes drift impossible at the literal level — both layers
import the same tuple object.
"""

from __future__ import annotations

from dqliteclient.connection import _is_no_tx_rollback_error
from dqlitewire import NO_TRANSACTION_MESSAGE_SUBSTRINGS


def test_recogniser_uses_wire_layer_substrings() -> None:
    """Drive the recogniser with each substring (alongside the right
    SQLite code) and verify it returns True. If a future maintainer
    drops a substring on either side, this test detects it via the
    wire constant."""
    from dqliteclient.exceptions import OperationalError

    for substr in NO_TRANSACTION_MESSAGE_SUBSTRINGS:
        exc = OperationalError(1, f"prefix {substr} suffix")
        assert _is_no_tx_rollback_error(exc), (
            f"recogniser must accept the substring {substr!r} "
            "from dqlitewire.NO_TRANSACTION_MESSAGE_SUBSTRINGS"
        )


def test_recogniser_rejects_unrelated_message() -> None:
    from dqliteclient.exceptions import OperationalError

    exc = OperationalError(1, "some unrelated SQLite error")
    assert not _is_no_tx_rollback_error(exc)


def test_substring_constant_has_expected_values() -> None:
    """Pin the canonical substring — the anchored
    ``"no transaction is active"`` is the single source of truth.
    Both upstream wordings (``"cannot rollback - no transaction is
    active"`` and ``"cannot commit - no transaction is active"``)
    contain this substring; the prior bare ``"cannot rollback"``
    token was removed because it was too permissive (any unrelated
    SQLite error or DQLITE_ERROR=1 message containing those words
    could trigger the silent-swallow path). The integration pin
    ``test_no_transaction_error_wording.py`` (in dbapi) catches a
    server-version drift against a live cluster."""
    assert "no transaction is active" in NO_TRANSACTION_MESSAGE_SUBSTRINGS
