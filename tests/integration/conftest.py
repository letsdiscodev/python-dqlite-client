"""Integration test fixtures for dqlite-client.

These tests require a running dqlite cluster.
Start the test cluster with:
    cd ../dqlite-test-cluster && docker compose up -d
"""

import os

import pytest

# Skip integration tests if cluster not available
DQLITE_TEST_CLUSTER = os.environ.get("DQLITE_TEST_CLUSTER", "localhost:9001")
# All cluster node addresses (host-mapped ports). Override via the
# ``DQLITE_TEST_CLUSTER_NODES`` env var (comma-separated). The default
# matches the dev fixed-port cluster (host ports 19001-19003).
DQLITE_TEST_CLUSTER_NODES = os.environ.get(
    "DQLITE_TEST_CLUSTER_NODES",
    "localhost:19001,localhost:19002,localhost:19003",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as requiring dqlite cluster")


@pytest.fixture
def cluster_address() -> str:
    """Get the test cluster address."""
    return DQLITE_TEST_CLUSTER


@pytest.fixture
def cluster_addresses() -> list[str]:
    """Get all test cluster addresses."""
    return ["localhost:9001", "localhost:9002", "localhost:9003"]


@pytest.fixture
def cluster_node_addresses() -> list[str]:
    """All host-mapped cluster node addresses, configurable via the
    ``DQLITE_TEST_CLUSTER_NODES`` env var. Use this for multi-node
    tests that need every node reachable (the older
    ``cluster_addresses`` fixture is hardcoded and kept for
    backwards-compatibility with the shared
    ``dqlite-test-cluster/pytest_fixtures.py`` shape)."""
    return [a.strip() for a in DQLITE_TEST_CLUSTER_NODES.split(",") if a.strip()]
