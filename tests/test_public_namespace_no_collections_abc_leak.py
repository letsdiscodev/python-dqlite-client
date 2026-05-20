"""Pin: ``collections.abc.Sequence`` is imported as a typing helper,
not as a public symbol. Underscore-alias keeps it out of the public
namespace, matching the sibling ``_Final`` convention.
"""

from __future__ import annotations

import dqliteclient


def test_collections_abc_sequence_not_publicly_exported() -> None:
    assert not hasattr(dqliteclient, "Sequence"), (
        "dqliteclient.Sequence leaks collections.abc.Sequence into the "
        "public namespace; use ``from collections.abc import Sequence as _Sequence``"
    )
