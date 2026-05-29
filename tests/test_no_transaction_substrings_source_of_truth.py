"""``_is_no_tx_rollback_error`` shares its substrings with the wire-layer constant."""

from __future__ import annotations

from dqliteclient.connection import _is_no_tx_rollback_error
from dqlitewire import NO_TRANSACTION_MESSAGE_SUBSTRINGS


def test_recogniser_uses_wire_layer_substrings() -> None:
    """Every wire-constant substring is recognised, detecting a drop on either side."""
    from dqliteclient.exceptions import OperationalError

    for substr in NO_TRANSACTION_MESSAGE_SUBSTRINGS:
        exc = OperationalError(f"prefix {substr} suffix", 1)
        assert _is_no_tx_rollback_error(exc), (
            f"recogniser must accept the substring {substr!r} "
            "from dqlitewire.NO_TRANSACTION_MESSAGE_SUBSTRINGS"
        )


def test_recogniser_rejects_unrelated_message() -> None:
    from dqliteclient.exceptions import OperationalError

    exc = OperationalError("some unrelated SQLite error", 1)
    assert not _is_no_tx_rollback_error(exc)


def test_substring_constant_has_expected_values() -> None:
    """The anchored ``"no transaction is active"`` is the canonical substring; the bare
    ``"cannot rollback"`` token was dropped as too permissive (matched unrelated errors)."""
    assert "no transaction is active" in NO_TRANSACTION_MESSAGE_SUBSTRINGS
