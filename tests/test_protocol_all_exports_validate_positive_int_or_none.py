"""Pin: ``dqliteclient.protocol`` lists ``validate_positive_int_or_none``
in ``__all__`` so ``from dqliteclient.protocol import *`` surfaces
the validator and IDE auto-import tooling treats it as a public
member of the submodule (not a private import).

The validator was already re-exported at the top-level package
(``dqliteclient.validate_positive_int_or_none``); this pins parity
on the submodule's own public-symbol declaration.
"""

from __future__ import annotations


def test_protocol_star_import_exposes_validate_positive_int_or_none() -> None:
    namespace: dict[str, object] = {}
    exec("from dqliteclient.protocol import *", namespace)
    assert "validate_positive_int_or_none" in namespace, (
        "dqliteclient.protocol.__all__ must list validate_positive_int_or_none "
        "so star-imports surface the public validator"
    )


def test_protocol_all_lists_validate_positive_int_or_none() -> None:
    import dqliteclient.protocol as protocol_mod

    assert "validate_positive_int_or_none" in protocol_mod.__all__
