"""Pin: ``collections.abc.Sequence`` is underscore-aliased so it does
not leak into ``dqliteclient``'s public namespace.
"""

from __future__ import annotations

import dqliteclient


def test_collections_abc_sequence_not_publicly_exported() -> None:
    assert not hasattr(dqliteclient, "Sequence"), (
        "dqliteclient.Sequence leaks collections.abc.Sequence into the "
        "public namespace; use ``from collections.abc import Sequence as _Sequence``"
    )
