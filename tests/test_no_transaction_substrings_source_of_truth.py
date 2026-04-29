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
    """Pin the canonical substring list — both clauses must be
    present so a server-version drift that drops one but not the
    other still triggers the recogniser. If a server change requires
    an update, the integration pin
    ``test_no_transaction_error_wording.py`` (in dbapi) catches it
    against a live cluster."""
    assert "no transaction is active" in NO_TRANSACTION_MESSAGE_SUBSTRINGS
    assert "cannot rollback" in NO_TRANSACTION_MESSAGE_SUBSTRINGS
