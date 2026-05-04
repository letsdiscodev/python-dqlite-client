"""Regression pin: ``pool.initialize``'s ``BaseExceptionGroup``
payload is bounded under hostile-peer fan-out. The group's display
message is a compact ``"pool.initialize: N of M connects failed"``
formatted string; the chained ``failures`` carry per-exception
``raw_message`` payloads that the ``OperationalError`` constructor
caps at 4 KiB. Net pickled-payload size scales as
``min_size × ~4 KiB`` rather than ``min_size × 64 KiB``.
"""

from __future__ import annotations

import pickle

from dqliteclient.exceptions import OperationalError


def test_operational_error_raw_message_capped_bounds_pickle_size() -> None:
    """A single OperationalError carrying a hostile 64 KiB server text
    pickles to ~5 KiB rather than ~70 KiB."""
    hostile = "X" * 64_000
    e = OperationalError(1, hostile, raw_message=hostile)
    pickled = pickle.dumps(e)
    # Empirically <10 KB; fail loudly if the cap regresses.
    assert len(pickled) < 10_000, (
        f"OperationalError pickle size {len(pickled)} bytes — raw_message cap regression"
    )


def test_aggregate_exception_group_payload_scales_with_min_size_not_unbounded() -> None:
    """A BaseExceptionGroup chain of N OperationalErrors each carrying
    capped raw_messages stays bounded at N × ~4 KiB rather than
    N × ~64 KiB. With min_size=10 against hostile peers, the group
    pickles to ~50 KB rather than ~640 KB."""
    hostile = "Y" * 64_000
    failures = [OperationalError(1, hostile, raw_message=hostile) for _ in range(10)]
    group = BaseExceptionGroup("pool.initialize: 10 of 10 connects failed", failures)
    pickled = pickle.dumps(group)
    assert len(pickled) < 100_000, (
        f"BaseExceptionGroup pickle size {len(pickled)} bytes — "
        f"per-exception raw_message cap not enforcing aggregate bound"
    )
