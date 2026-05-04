"""``DqliteConnection._validate_params`` rejects single-shot iterators
(generators, zip, map, filter, ...) with a friendly ``DataError`` at
the call site rather than letting the wire encoder fail with
``TypeError: object of type 'generator' has no len()`` deep inside
``ExecSqlRequest.__post_init__``.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError


@pytest.mark.parametrize(
    "factory",
    [
        lambda: (x for x in [1, 2, 3]),
        lambda: zip([1, 2], [3, 4], strict=True),
        lambda: map(int, ["1", "2"]),
        lambda: filter(None, [1, 2, 3]),
        lambda: iter([1, 2, 3]),
    ],
    ids=[
        "generator",
        "zip",
        "map",
        "filter",
        "list-iter",
    ],
)
def test_validate_params_rejects_single_shot_iterators(factory) -> None:
    with pytest.raises(DataError, match="single-shot iterators"):
        DqliteConnection._validate_params(factory())


def test_validate_params_accepts_list_and_tuple() -> None:
    DqliteConnection._validate_params([1, 2, 3])
    DqliteConnection._validate_params((1, 2, 3))
    DqliteConnection._validate_params([])
    DqliteConnection._validate_params(())


def test_validate_params_accepts_user_sequence() -> None:
    """A class implementing ``__len__`` and ``__getitem__`` is an
    explicit Sequence; let it through (matches the duck-type
    behaviour of the wire encoder)."""

    class MySeq:
        def __init__(self, data: list[int]) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> int:
            return self._data[idx]

    DqliteConnection._validate_params(MySeq([1, 2, 3]))


def test_validate_params_accepts_none() -> None:
    DqliteConnection._validate_params(None)
