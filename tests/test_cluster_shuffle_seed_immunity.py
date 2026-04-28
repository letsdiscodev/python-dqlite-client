"""Pin: cluster shuffle is immune to ``random.seed()``.

The cluster module uses ``random.SystemRandom`` (exposed as
``_cluster_random``) so that a downstream call to ``random.seed(...)``
— common in test suites or in libraries that "want deterministic
behavior" without scoping it — cannot defeat the per-sweep
stampede-avoidance shuffle.

If we accidentally regress to module-level ``random.shuffle``, the
seeded test below would produce identical shuffles every run and
this test would fail.
"""

from __future__ import annotations

import random

from dqliteclient.cluster import _cluster_random


def test_cluster_random_is_system_random() -> None:
    """The cluster shuffle source is a SystemRandom instance, not the
    module-level Random."""
    assert isinstance(_cluster_random, random.SystemRandom)


def test_cluster_shuffle_unaffected_by_seed() -> None:
    """Two shuffles taken with identical seeds applied to the global
    PRNG must still differ — proving the cluster shuffle does not
    consume the global PRNG state."""
    nodes_first: list[int] = list(range(100))
    nodes_second: list[int] = list(range(100))

    random.seed(42)
    _cluster_random.shuffle(nodes_first)

    random.seed(42)
    _cluster_random.shuffle(nodes_second)

    # With SystemRandom, two seeded global states do not produce the
    # same shuffle. (The probability of a 100-element collision is
    # 1/100! — astronomically low.)
    assert nodes_first != nodes_second, (
        "cluster shuffle must use OS entropy, not the seeded module PRNG"
    )


def test_cluster_shuffle_does_not_consume_module_random_state() -> None:
    """A cluster shuffle in between two module-level ``random.random()``
    draws must not affect the second draw — proving the cluster
    shuffle is isolated from the module PRNG.
    """
    random.seed(123)
    _ = random.random()
    expected_next = random.random()

    # Reset, take one draw, then a cluster shuffle, then the next draw.
    random.seed(123)
    _ = random.random()
    _cluster_random.shuffle(list(range(100)))
    actual_next = random.random()

    assert actual_next == expected_next, (
        "cluster shuffle must not consume bytes from the global module-level Random instance"
    )
