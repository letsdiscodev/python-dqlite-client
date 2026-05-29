"""``DqliteConnection._validate_params`` rejects single-shot iterators with a
``DataError`` rather than a cryptic no-len TypeError deep in the wire encoder."""

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
    """A class with ``__len__`` and ``__getitem__`` is a Sequence; allow it."""

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
