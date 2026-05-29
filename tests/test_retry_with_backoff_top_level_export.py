"""``retry_with_backoff`` is re-exported at the top level, reachable as
``dqliteclient.<name>`` like the sibling utility helpers."""

import dqliteclient
from dqliteclient.retry import retry_with_backoff as submodule_helper


def test_retry_with_backoff_callable() -> None:
    assert callable(dqliteclient.retry_with_backoff)


def test_retry_with_backoff_in_all() -> None:
    assert "retry_with_backoff" in dqliteclient.__all__


def test_retry_with_backoff_identity_with_submodule() -> None:
    assert dqliteclient.retry_with_backoff is submodule_helper
