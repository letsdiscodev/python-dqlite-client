"""Pin: ``pool.initialize``'s ``BaseExceptionGroup`` payload stays bounded
under hostile-peer fan-out because ``OperationalError`` caps each
``raw_message`` at 4 KiB — pickled size scales as min_size × ~4 KiB."""

from __future__ import annotations

import pickle

from dqliteclient.exceptions import OperationalError


def test_operational_error_raw_message_capped_bounds_pickle_size() -> None:
    """A 64 KiB raw_message pickles to ~5 KiB rather than ~70 KiB."""
    hostile = "X" * 64_000
    e = OperationalError(hostile, 1, raw_message=hostile)
    pickled = pickle.dumps(e)
    assert len(pickled) < 10_000, (
        f"OperationalError pickle size {len(pickled)} bytes — raw_message cap regression"
    )


def test_aggregate_exception_group_payload_scales_with_min_size_not_unbounded() -> None:
    """A group of 10 capped OperationalErrors pickles to ~50 KB, not ~640 KB."""
    hostile = "Y" * 64_000
    failures = [OperationalError(hostile, 1, raw_message=hostile) for _ in range(10)]
    group = BaseExceptionGroup("pool.initialize: 10 of 10 connects failed", failures)
    pickled = pickle.dumps(group)
    assert len(pickled) < 100_000, (
        f"BaseExceptionGroup pickle size {len(pickled)} bytes — "
        f"per-exception raw_message cap not enforcing aggregate bound"
    )
