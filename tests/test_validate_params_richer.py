"""``DqliteConnection._validate_params`` rejects scalar-iterable and
unordered containers.

Previously only ``str | bytes`` were rejected, leaving ``bytearray``,
``memoryview``, ``Mapping``, and ``set`` / ``frozenset`` as silent
footguns at the bind layer — they iterate in ways that scramble the
positional qmark binding. Match the richer dbapi-layer
``_reject_non_sequence_params`` validator so callers that go direct
to the client layer are not silently exposed.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError


def _make_conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestClientValidateParamsRichRejections:
    def test_rejects_bytearray(self) -> None:
        with pytest.raises(DataError, match="bytearray"):
            _make_conn()._validate_params(bytearray(b"abc"))

    def test_rejects_memoryview(self) -> None:
        with pytest.raises(DataError, match="memoryview"):
            _make_conn()._validate_params(memoryview(b"abc"))

    def test_rejects_dict(self) -> None:
        with pytest.raises(DataError, match="mapping"):
            _make_conn()._validate_params({"a": 1})  # type: ignore[arg-type]

    def test_rejects_set(self) -> None:
        with pytest.raises(DataError, match="set"):
            _make_conn()._validate_params({1, 2, 3})  # type: ignore[arg-type]

    def test_rejects_frozenset(self) -> None:
        with pytest.raises(DataError, match="set"):
            _make_conn()._validate_params(frozenset({1, 2, 3}))  # type: ignore[arg-type]

    def test_str_and_bytes_still_rejected(self) -> None:
        with pytest.raises(DataError, match="str"):
            _make_conn()._validate_params("abc")  # type: ignore[arg-type]
        with pytest.raises(DataError, match="bytes"):
            _make_conn()._validate_params(b"abc")  # type: ignore[arg-type]

    def test_list_and_tuple_accepted(self) -> None:
        _make_conn()._validate_params([1, 2])
        _make_conn()._validate_params((1, 2))

    def test_none_accepted(self) -> None:
        _make_conn()._validate_params(None)
