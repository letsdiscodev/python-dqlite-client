"""``dqliteclient.protocol.__all__`` lists ``validate_positive_int_or_none``
so star-imports and IDE tooling treat it as a public submodule member."""

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
