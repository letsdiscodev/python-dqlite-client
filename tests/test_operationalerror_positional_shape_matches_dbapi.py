"""Pin: ``client.OperationalError`` and ``dbapi.OperationalError``
share the same ``(message, code)`` positional shape.

The two packages export classes with the same name; both code-bearing.
Cross-package bridges (SA dialect ``is_disconnect``, retry middleware,
custom decorators) commonly pass args positionally between them.
Without alignment, a passthrough silently swaps fields:
``code`` becomes a string, defeating every code-based check downstream.
"""

from __future__ import annotations

import dqliteclient.exceptions as ce


def test_client_operationalerror_positional_args_match_message_code() -> None:
    """``OperationalError("message text", 42)`` — message first, code second.
    Mirrors stdlib ``sqlite3.Error`` and ``dqlitedbapi.OperationalError``."""
    e = ce.OperationalError("boom", 42)
    assert e.message == "boom"
    assert e.code == 42


def test_client_operationalerror_with_raw_message_kwarg() -> None:
    e = ce.OperationalError("display", 5, raw_message="full server text")
    assert e.message == "display"
    assert e.code == 5
    assert e.raw_message == "full server text"


def test_pickle_round_trip_preserves_positional_shape() -> None:
    """``__reduce__`` returns ``(cls, self.args, ...)``. After the
    positional flip, ``self.args == (message, code)``; pickle must
    reconstruct via ``OperationalError(message, code)``."""
    import pickle

    e = ce.OperationalError("constraint failed", 19, raw_message="full text")
    restored = pickle.loads(pickle.dumps(e))
    assert restored.message == "constraint failed"
    assert restored.code == 19
    assert restored.raw_message == "full text"
