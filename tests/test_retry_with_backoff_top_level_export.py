"""``retry_with_backoff`` is part of the public surface — same
posture as ``parse_address`` / ``validate_positive_int_or_none``.

The retry submodule was the source of truth (its ``__all__``
declared the helper public) but ``__init__.py`` did not re-export.
Downstream callers wanting the same retry policy as the client
itself had to import from the submodule path; this aligns with the
sibling utility helpers reachable as ``dqliteclient.<name>``.
"""

import dqliteclient
from dqliteclient.retry import retry_with_backoff as submodule_helper


def test_retry_with_backoff_callable() -> None:
    assert callable(dqliteclient.retry_with_backoff)


def test_retry_with_backoff_in_all() -> None:
    assert "retry_with_backoff" in dqliteclient.__all__


def test_retry_with_backoff_identity_with_submodule() -> None:
    assert dqliteclient.retry_with_backoff is submodule_helper
