"""``client.OperationalError`` and ``dbapi.OperationalError`` share the same
``(message, code)`` positional shape so cross-package passthroughs don't swap fields."""

from __future__ import annotations

import dqliteclient.exceptions as ce


def test_client_operationalerror_positional_args_match_message_code() -> None:
    """Message first, code second, mirroring stdlib sqlite3 and dqlitedbapi."""
    e = ce.OperationalError("boom", 42)
    assert e.message == "boom"
    assert e.code == 42


def test_client_operationalerror_with_raw_message_kwarg() -> None:
    e = ce.OperationalError("display", 5, raw_message="full server text")
    assert e.message == "display"
    assert e.code == 5
    assert e.raw_message == "full server text"


def test_pickle_round_trip_preserves_positional_shape() -> None:
    """``__reduce__`` relies on ``self.args == (message, code)`` for round-trip."""
    import pickle

    e = ce.OperationalError("constraint failed", 19, raw_message="full text")
    restored = pickle.loads(pickle.dumps(e))
    assert restored.message == "constraint failed"
    assert restored.code == 19
    assert restored.raw_message == "full text"
